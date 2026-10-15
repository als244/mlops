"""FP8 expert math for the SonicMoE implementations (``"fp8_current"``).

The expert weights arrive quantized: `QuackFP8Weight` containers
(``expert_parallel/quack/parameters/fp8.py``) holding each weight's E4M3 rows
and E4M3 transpose with a descale per row of each, built once -- from a
checkpoint, or published from bf16/fp32 masters after an optimizer step (a
bf16 weight asked for FP8 math is quantized per call the same way, straight
through for its gradient). ``w13`` keeps its rows interleaved (``gate_j``,
``up_j`` adjacent), the order Quack's paired SwiGLU epilogue reads, so the up
projection needs no concatenated layout and the backward's ``dh`` no
reordering. The routed rows, the activation and the cotangents are quantized
per row per call (current scaling) by the expert-parallel path's kernels,
gathered or activated in the same pass. The GEMMs are that path's
scale-aware epilogue kernels (``expert_parallel/quack/experts/fp8.py``,
``wgrad.py``), varlen over this provider's expert offsets. Weight gradients
stay the bf16 forms' varlen-K GEMMs, in the containers' logical dtype and
``nn.Linear`` row order (SM90's varlen-K kernels have no FP8 layout).

Inside the custom operators a container is its components, a `Weight`.

Block-scaled weights (`BlockFP8Weight`, ``compute_precision="fp8_block"``)
come through the same functions: each takes a `block_fp8.BlockWeight` to that
module's forms, whose one stored copy serves both orientations.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import NamedTuple

import torch

from ...block_fp8 import BlockFP8Weight
from ...dispatch.registry import SupportResult
from ...expert_parallel.quack.parameters.fp8 import QuackFP8Weight
from . import block_fp8
from .block_fp8 import BlockWeight

# The expert GEMMs' FP8 precisions: current per-row scaling on two-copy
# `QuackFP8Weight` containers, or 128x128 blocks on one-copy `BlockFP8Weight`s.
PRECISIONS = ("fp8_current", "fp8_block")
CONTAINERS = {"fp8_current": QuackFP8Weight, "fp8_block": BlockFP8Weight}

CAPABILITY = 9  # the FP8 expert kernels are validated on SM90 only
ALIGNMENT = 16  # elements per 16-byte row pitch of E4M3 operands


class Weight(NamedTuple):
    """A container's components: the operands of one expert weight."""

    data: torch.Tensor  # E4M3 [E, out, in], rows as the forward reads them
    transposed: torch.Tensor  # E4M3 [E, in, out], the backward's K-major Wᵀ
    scale: torch.Tensor  # FP32 [E, out] descales of ``data``'s rows
    transposed_scale: torch.Tensor  # FP32 [E, in] descales of ``transposed``'s rows

    @property
    def shape(self):
        return self.data.shape


def available() -> bool:
    return all(find_spec(name) is not None for name in ("quack", "triton"))


def is_container(weight) -> bool:
    return isinstance(weight, (QuackFP8Weight, BlockFP8Weight))


def supports(h2, w13_experts, w2_experts, precision="fp8_current") -> SupportResult:
    """Whether FP8 math at ``precision`` applies, after the bf16 form's own
    checks: the containers are that precision's (plain weights are quantized
    per call); for ``"fp8_current"`` SM90, the extras, 16-element widths, and
    ``w13`` interleaved when it is a container; ``"fp8_block"`` as
    `block_fp8.supports` says."""
    for weight in (w13_experts, w2_experts):
        if is_container(weight) and not isinstance(weight, CONTAINERS[precision]):
            return SupportResult.no(
                f"{type(weight).__name__} weights do not compute at {precision}; "
                "ask compute_precision='bf16' to dequantize them"
            )
    if precision == "fp8_block":
        return block_fp8.supports(h2, w13_experts, w2_experts)
    if not h2.is_cuda or torch.cuda.get_device_capability(h2.device)[0] != CAPABILITY:
        return SupportResult.no(
            "fp8_current expert math is served on compute capability 9.x"
        )
    if not available():
        return SupportResult.no("fp8_current needs the quack extra and triton")
    if h2.shape[-1] % ALIGNMENT or w2_experts.shape[2] % ALIGNMENT:
        return SupportResult.no(
            "fp8_current needs model and expert widths that are multiples of 16"
        )
    if is_container(w13_experts) and not w13_experts.interleaved:
        return SupportResult.no(
            "an FP8 w13 must be quantized with interleaved=True (gate/up rows adjacent)"
        )
    return SupportResult.yes()


def resolve(compute_precision, w13_experts, w2_experts) -> str:
    """The forms' precision for a request: an FP8 precision when asked, or by
    default the containers' own (``"fp8_block"`` for `BlockFP8Weight`,
    ``"fp8_current"`` for `QuackFP8Weight`); ``"native"`` (bf16) otherwise."""
    if compute_precision in PRECISIONS:
        return compute_precision
    if compute_precision is None:
        weights = (w13_experts, w2_experts)
        if any(isinstance(weight, BlockFP8Weight) for weight in weights):
            return "fp8_block"
        if any(isinstance(weight, QuackFP8Weight) for weight in weights):
            return "fp8_current"
    return "native"


def containers(w13_experts, w2_experts, precision="fp8_current"):
    """The weights as ``precision``'s containers, quantizing a plain weight
    per call."""
    if precision == "fp8_block":
        return tuple(
            weight if is_container(weight) else BlockFP8Weight.quantize(weight)
            for weight in (w13_experts, w2_experts)
        )
    if not is_container(w13_experts):
        w13_experts = QuackFP8Weight.quantize(w13_experts, interleaved=True)
    if not is_container(w2_experts):
        w2_experts = QuackFP8Weight.quantize(w2_experts)
    return w13_experts, w2_experts


def dequantized(w13_experts, w2_experts, *, dtype):
    """The weights as plain tensors at ``dtype``, containers dequantized."""
    return tuple(
        weight.dequantize(dtype=dtype) if is_container(weight) else weight
        for weight in (w13_experts, w2_experts)
    )


def prepare(compute_precision, w13_experts, w2_experts, dtype):
    """At apply time: ``(precision, w13, w2, weight_grad_dtype)`` -- containers
    for FP8 math (plain weights quantized per call) with the gradient dtype
    policy resolved to the containers' logical dtype, or plain weights at
    ``dtype`` for bf16 math (containers dequantized per call)."""
    from ...dispatch.context import weight_gradient_dtype

    precision = resolve(compute_precision, w13_experts, w2_experts)
    grad_dtype = weight_gradient_dtype()
    if precision in PRECISIONS:
        w13_experts, w2_experts = containers(w13_experts, w2_experts, precision)
        if grad_dtype is None:
            grad_dtype = w13_experts.dtype
    else:
        w13_experts, w2_experts = dequantized(w13_experts, w2_experts, dtype=dtype)
    return precision, w13_experts, w2_experts, grad_dtype


def weights(compute_precision, components, *data):
    """Inside an operator: for FP8 math, each E4M3 ``data`` with its further
    components (``weight_components``, in weight order) as a `Weight` (three
    each, ``"fp8_current"``) or a `BlockWeight` (one, ``"fp8_block"``); the
    tensors unchanged otherwise."""
    if compute_precision not in PRECISIONS:
        return data
    kind, count = (
        (Weight, 3) if compute_precision == "fp8_current" else (BlockWeight, 1)
    )
    if len(components) != count * len(data):
        raise ValueError(
            f"{compute_precision} operators take {count} further components per "
            f"weight, got {len(components)} for {len(data)} weights"
        )
    return tuple(
        kind(value, *components[count * index : count * index + count])
        for index, value in enumerate(data)
    )


def weight_of(components, weight):
    """Inside a one-weight operator: ``weight`` with its components as the
    form they name (three: a `Weight`, one: a `BlockWeight`), or unchanged
    without any."""
    if not components:
        return weight
    (value,) = weights(
        "fp8_current" if len(components) == 3 else "fp8_block", components, weight
    )
    return value


def register_container_dispatch(*operators):
    """Let the operators take `QuackFP8Weight` arguments, as the expert-parallel
    operators do: each container becomes its E4M3 rows in place and its other
    components fill the trailing ``weight_components`` list, in weight order,
    before the operator runs on plain tensors."""

    def handler(cls, func, types, args, kwargs):
        del cls, types
        if kwargs:
            raise ValueError("Use positional operator arguments")
        args = list(args)
        components = []
        for index, value in enumerate(args):
            if isinstance(value, QuackFP8Weight):
                args[index] = value._data
                components.extend(
                    (value._transposed, value._scale, value._transposed_scale)
                )
            elif isinstance(value, BlockFP8Weight):
                args[index] = value._data
                components.append(value._scale)
        # The ``weight_components`` slot is the operator's last list argument
        # (``lengths`` comes before it), empty at the logical call.
        slot = max(index for index, value in enumerate(args) if isinstance(value, list))
        args[slot] = components
        return func(*args)

    for operator in operators:
        for container in CONTAINERS.values():
            operator.register_torch_dispatch(container, handler)


def _kernels():
    from ...expert_parallel.quack.experts.fp8 import (
        scaled_swiglu,
        scaled_swiglu_backward,
    )
    from ...expert_parallel.quack.experts.wgrad import scaled_fp8
    from ...expert_parallel.quack.kernels.quantize_rows import quantize_rows_fp8
    from ...expert_parallel.quack.kernels.swiglu_rows import swiglu_quantize_rows_fp8

    return (
        scaled_swiglu,
        scaled_swiglu_backward,
        scaled_fp8,
        quantize_rows_fp8,
        swiglu_quantize_rows_fp8,
    )


def up_projection(h2_flat, w13: Weight, offsets, gather, limit=None):
    """Gather the routed rows, project through ``w13`` in FP8 and apply SwiGLU,
    clamped at ``limit`` when given: returns ``(h, a)``, the interleaved
    pre-activation and the activation."""
    if isinstance(w13, BlockWeight):
        return block_fp8.up_projection(h2_flat, w13, offsets, gather, limit)
    scaled_swiglu, _, _, quantize_rows, _ = _kernels()
    rows, scales = quantize_rows(h2_flat, gather=gather)
    options = {}
    if limit is not None:
        from ...expert_parallel.quack.experts.swiglu_limit import scaled_swiglu_limit

        scaled_swiglu, options = scaled_swiglu_limit, {"limit": float(limit)}
    result = scaled_swiglu(
        rows,
        w13.data.transpose(1, 2),
        xs=scales,
        ws=w13.scale,
        out_dtype=h2_flat.dtype,
        cu_seqlens_m=offsets,
        **options,
    )
    return result["D"], result["postact"]


def _project_down(rows, scales, w2: Weight, offsets, dtype):
    _, _, scaled_fp8, _, _ = _kernels()
    y = torch.empty(rows.shape[0], w2.data.shape[1], dtype=dtype, device=rows.device)
    scaled_fp8(
        rows,
        w2.data.transpose(1, 2),
        out={"D": y},
        xs=scales,
        ws=w2.scale,
        cu_seqlens_m=offsets,
    )
    return y


def project_down(a, w2: Weight, offsets):
    """``y = a @ w2ᵀ`` per expert in FP8 from the activation rows."""
    if isinstance(w2, BlockWeight):
        return block_fp8.project_down(a, w2, offsets)
    _, _, _, quantize_rows, _ = _kernels()
    rows, scales = quantize_rows(a)
    return _project_down(rows, scales, w2, offsets, a.dtype)


def project_down_from_preactivation(h, w2: Weight, offsets, limit=None):
    """The same from the interleaved pre-activation: SwiGLU, clamped at
    ``limit`` when given, and the row quantization in one pass (the composed
    form's rematerialization)."""
    if isinstance(w2, BlockWeight):
        return block_fp8.project_down_from_preactivation(h, w2, offsets, limit)
    _, _, _, _, swiglu_quantize_rows = _kernels()
    rows, scales = swiglu_quantize_rows(h, limit=limit)
    return _project_down(rows, scales, w2, offsets, h.dtype)


def down_backward_activations(
    grad_output_flat, h, w2: Weight, score_rows, offsets, gather, limit=None
):
    """One FP8 GEMM with the SwiGLU backward in its epilogue, clamped at
    ``limit`` when given: returns ``(dh, a', dscore_rows)``, the interleaved
    pre-activation gradient, the route-scaled activation and each sorted
    row's score gradient."""
    if isinstance(w2, BlockWeight):
        return block_fp8.down_backward_activations(
            grad_output_flat, h, w2, score_rows, offsets, gather, limit
        )
    _, scaled_swiglu_backward, _, quantize_rows, _ = _kernels()
    rows, scales = quantize_rows(grad_output_flat, gather=gather)
    options = {}
    if limit is not None:
        from ...expert_parallel.quack.experts.swiglu_limit import (
            scaled_dswiglu_limit_scored,
        )

        scaled_swiglu_backward = scaled_dswiglu_limit_scored
        options = {"limit": float(limit)}
    result = scaled_swiglu_backward(
        rows,
        w2.transposed.transpose(1, 2),
        C=h,
        xs=scales,
        ws=w2.transposed_scale,
        score=score_rows,
        cu_seqlens_m=offsets,
        out_dtype=h.dtype,
        **options,
    )
    return result["D"], result["postact"], result["dscore"]


def input_gradient_rows(cotangent, weight: Weight, offsets, gather=None, gate_up=False):
    """``rows @ W`` per expert in FP8 through the weight's K-major transpose:
    ``dx = dh @ w13`` (the interleaved ``dh`` against the same interleaving of
    ``w13ᵀ``'s K order) or ``da = dy @ w2``; the rows gathered from
    ``cotangent`` by ``gather`` as they are quantized, when given.
    ``gate_up`` names ``w13``, which a block-scaled weight holds in gate-then-up
    order."""
    if isinstance(weight, BlockWeight):
        return block_fp8.input_gradient_rows(
            cotangent, weight, offsets, gather=gather, gate_up=gate_up
        )
    _, _, scaled_fp8, quantize_rows, _ = _kernels()
    rows, scales = quantize_rows(cotangent, gather=gather)
    out = torch.empty(
        rows.shape[0],
        weight.transposed.shape[1],
        dtype=cotangent.dtype,
        device=cotangent.device,
    )
    scaled_fp8(
        rows,
        weight.transposed.transpose(1, 2),
        out={"D": out},
        xs=scales,
        ws=weight.transposed_scale,
        cu_seqlens_m=offsets,
    )
    return out


@lru_cache(maxsize=1)
def _lora_up_epilogue():
    from quack.activation import swiglu
    from quack.epilogue.frontend import gemm_epilogue
    from quack.epilogue.math import unpack
    from quack.epilogue.ops import ColVecLoad, RowVecLoad

    @gemm_epilogue(
        outputs=("postact",),
        mode="acc_pair",
        ops={"xs": ColVecLoad("xs"), "ws": RowVecLoad("ws")},
    )
    def scaled_swiglu_add(acc, c, xs, ws):
        value = acc * xs * ws + c
        gate, up = unpack(value)
        return {"D": value, "postact": swiglu(gate, up)}

    return scaled_swiglu_add


@lru_cache(maxsize=1)
def _lora_down_epilogue():
    from quack.epilogue.frontend import gemm_epilogue
    from quack.epilogue.ops import ColVecLoad, RowVecLoad

    @gemm_epilogue(ops={"xs": ColVecLoad("xs"), "ws": RowVecLoad("ws")})
    def scaled_fp8_add(acc, c, xs, ws):
        return {"D": acc * xs * ws + c}

    return scaled_fp8_add


def lora_down_projection(a, w2: Weight, c, offsets):
    """``y = a @ w2ᵀ + c`` per expert in FP8, ``c`` the LoRA term."""
    if isinstance(w2, BlockWeight):
        return block_fp8.lora_down_projection(a, w2, c, offsets)
    _, _, _, quantize_rows, _ = _kernels()
    rows, scales = quantize_rows(a)
    y = torch.empty(rows.shape[0], w2.data.shape[1], dtype=a.dtype, device=a.device)
    _lora_down_epilogue()(
        rows,
        w2.data.transpose(1, 2),
        C=c,
        out={"D": y},
        xs=scales,
        ws=w2.scale,
        cu_seqlens_m=offsets,
    )
    return y


def lora_up_projection(x, w13: Weight, c, offsets, gather=None, limit=None):
    """``h = x @ w13ᵀ + c`` per expert in FP8 with SwiGLU in the epilogue,
    clamped at ``limit`` when given, ``c`` the LoRA term in the interleaved
    column order: ``(h, a)``; the rows gathered from ``x`` by ``gather`` as
    they are quantized, when given."""
    if isinstance(w13, BlockWeight):
        return block_fp8.lora_up_projection(x, w13, c, offsets, gather, limit)
    _, _, _, quantize_rows, _ = _kernels()
    rows, scales = quantize_rows(x, gather=gather)
    epilogue, options = _lora_up_epilogue(), {}
    if limit is not None:
        from ...expert_parallel.quack.experts.swiglu_limit import (
            scaled_swiglu_limit_add,
        )

        epilogue, options = scaled_swiglu_limit_add, {"limit": float(limit)}
    result = epilogue(
        rows,
        w13.data.transpose(1, 2),
        C=c,
        xs=scales,
        ws=w13.scale,
        out_dtype=x.dtype,
        cu_seqlens_m=offsets,
        **options,
    )
    return result["D"], result["postact"]
