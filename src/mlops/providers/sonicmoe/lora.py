"""Expert LoRA on SonicMoE's kernels, for ``mlops.lora.experts.expert_lora``.

The composition keeps ``expert_lora``'s trace-visible routing (the package's
router with its own backward) and SonicMoE's routing metadata, and runs
everything else as two opaque operators that take a base weight with its two
low-rank factors, the structure of the expert-parallel Quack LoRA math
(``expert_parallel/quack/lora/math.py``) on the plain SonicMoE path's fused
pieces:

* up: the routed rows are never materialized -- the rank projection
  ``z = x @ Aᵀ`` and the base GEMM gather them from the token rows through
  Quack's ``A_idx`` -- and ``c = z @ Bᵀ`` enters the base GEMM before SwiGLU
  through ``gemm_act``'s ``C`` input. Its differentiable output is the
  interleaved pre-activation ``h``; the activation comes out of the same
  epilogue for the down projection and is not saved. The backward takes
  ``h``'s cotangent ``dh``, forms ``dz = dh @ B``, accumulates the LoRA
  branch ``dz @ A`` into the base's row gradient in place
  (``gemm_add_inplace``), sums the rows back onto their tokens with
  SonicMoE's token-broadcast backward, and takes ``dA = dzᵀ x`` and
  ``dB = dhᵀ z`` as varlen-K GEMMs, ``x`` gathered by the same index;
* down: ``y = a @ Wᵀ + z @ Bᵀ`` through ``gemm_add``, then SonicMoE's router
  kernel combines the rows onto their tokens with the route weights and the
  residual. The backward reads the token cotangent through ``A_idx`` into the
  unscaled row gradient ``r = dy @ W`` (LoRA branch accumulated in place) and
  one row pass over ``r`` and ``h`` gives the score gradient ``<r, a>``,
  the pre-activation gradient ``dh`` of ``score · r`` and, when a weight or
  factor gradient needs it, ``a' = score · a``.

So no ``[rows, H]`` gather or scatter runs outside a GEMM, no LoRA term is
added by a separate pass, no branch gradient is summed by autograd, and the
activation is neither saved nor re-read. The LoRA scale is folded into the
small ``B`` factors by the composition, which hands ``B`` for the up
projection with its gate/up rows interleaved, the order of ``h``. Base
weights get weight gradients only when they require them; frozen factors
cost no factor-gradient GEMM.

FP8 base weights (``QuackFP8Weight`` containers, or ``compute_precision=
"fp8_current"``) run the two heavy projections and their row gradients as
``fp8.py``'s scale-aware GEMMs, the rows gathered by the quantizers and the
LoRA terms added in the epilogues the same way, the operators taking the
containers as the ``moe`` operators do.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ...dispatch.context import weight_gradient_dtype
from ...dispatch.costs import flop_formula
from ...dispatch.registry import SupportResult
from ...expert_parallel.quack.kernels.swiglu_rows import routed_swiglu_rows_backward
from ...lora.routing import routing
from . import fp8
from .moe import CAPABILITIES, ROW_ALIGNMENT, available, kernels, metadata


def supports(hidden, gate_up, down, *, compute_precision=None) -> SupportResult:
    """Whether the SonicMoE expert-LoRA path takes these tensors, at
    ``compute_precision`` as `mlops.moe` reads it."""
    if not hidden.is_cuda:
        return SupportResult.no("SonicMoE runs on CUDA devices")
    if torch.cuda.get_device_capability(hidden.device)[0] not in CAPABILITIES:
        return SupportResult.no("SonicMoE serves compute capability 9.x and 10.x")
    if not available():
        return SupportResult.no("the quack and sonicmoe extras are not installed")
    plain = [t for t in (gate_up, down) if not fp8.is_container(t)]
    if any(t.dtype is not torch.bfloat16 for t in (hidden, *plain)):
        return SupportResult.no("SonicMoE computes in bfloat16")
    if hidden.shape[-1] % ROW_ALIGNMENT or down.shape[2] % ROW_ALIGNMENT:
        return SupportResult.no("the model and expert widths must be multiples of 8")
    if not (gate_up.is_contiguous() and down.is_contiguous()):
        return SupportResult.no("expert weights must be contiguous")
    if compute_precision not in (None, "bf16", "fp8_current"):
        return SupportResult.no(f"unknown compute precision {compute_precision!r}")
    if fp8.resolve(compute_precision, gate_up, down) == "fp8_current":
        return fp8.supports(hidden, gate_up, down)
    return SupportResult.yes()


def _weight(components, weight):
    """The operator's weight: a `fp8.Weight` when components came along."""
    (weight,) = fp8.weights(
        "fp8_current" if components else "native", components, weight
    )
    return weight


def interleaved_rows(b):
    """``[E, 2I, r]`` with gate rows then up rows -> gate/up rows alternating."""
    experts, packed, rank = b.shape
    return b.view(experts, 2, packed // 2, rank).transpose(1, 2).reshape(b.shape)


def _low(x, factor, offsets, gather=None):
    """``x @ factor[e]ᵀ`` over expert-sorted rows, the rows gathered from
    ``x`` by ``gather`` when given: a rank projection."""
    from quack.gemm_interface import gemm

    rows = x.shape[0] if gather is None else gather.shape[0]
    out = torch.empty(rows, factor.shape[1], dtype=x.dtype, device=x.device)
    gemm(x, factor.transpose(1, 2), out=out, cu_seqlens_m=offsets, A_idx=gather)
    return out


def _through(rows, factor, offsets, gather=None):
    """``rows @ factor[e]`` over expert-sorted rows (the factor as stored),
    the rows gathered by ``gather`` when given."""
    from quack.gemm_interface import gemm

    count = rows.shape[0] if gather is None else gather.shape[0]
    out = torch.empty(count, factor.shape[2], dtype=rows.dtype, device=rows.device)
    gemm(rows, factor, out=out, cu_seqlens_m=offsets, A_idx=gather)
    return out


def _accumulate_through(rows, factor, out, offsets):
    """``out += rows @ factor[e]`` in place: the LoRA branch joining the base's
    row gradient without a separate add."""
    from quack.gemm_interface import gemm_add_inplace

    gemm_add_inplace(rows, factor, out=out, cu_seqlens_m=offsets)


def _per_expert_product(left, right, out, offsets, gather=None, concat=None):
    """``out[e] = left[e]ᵀ @ right[e]`` over each expert's rows (varlen-K),
    ``left``'s rows gathered by ``gather`` when given."""
    from quack.gemm_interface import gemm

    options = {} if concat is None else {"concat_layout": concat}
    gemm(
        left.T,
        right,
        out=out,
        cu_seqlens_k=offsets,
        A_idx=gather,
        batch_idx_permute=None,
        dynamic_scheduler=False,
        **options,
    )
    return out


def _empty(shape, dtype, device):
    return torch.empty(tuple(shape), dtype=dtype, device=device)


@torch.library.custom_op("mlops::lora_sonicmoe_up_fwd", mutates_args=())
def _up_forward(
    x: torch.Tensor,
    gather: torch.Tensor,
    reverse: torch.Tensor,
    gate_up: torch.Tensor,
    a13: torch.Tensor,
    b13: torch.Tensor,
    offsets: torch.Tensor,
    top_k: int,
    grad_dtype: torch.dtype | None,
    need_x: bool,
    need_weight: bool,
    need_factors: bool,
    weight_components: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Over the expert-sorted rows ``x[gather]``: ``z = x @ a13ᵀ``,
    ``h = x @ gate_upᵀ + z @ b13ᵀ`` and ``a = swiglu(h)``, as ``(h, a, z)``;
    ``h`` in Quack's interleaved gate/up column order (``b13``'s rows come
    interleaved and scaled). ``reverse`` and ``top_k`` serve the backward."""
    from quack.gemm_interface import gemm_act

    kernels()
    z = _low(x, a13, offsets, gather)
    c = _low(z, b13, offsets)
    if weight_components:
        h, a = fp8.lora_up_projection(
            x, _weight(weight_components, gate_up), c, offsets, gather=gather
        )
        return h, a, z
    rows, width = gather.shape[0], gate_up.shape[1] // 2
    h = torch.empty(rows, 2 * width, dtype=x.dtype, device=x.device)
    a = torch.empty(rows, width, dtype=x.dtype, device=x.device)
    gemm_act(
        x,
        gate_up.transpose(1, 2),
        C=c,
        activation="swiglu",
        cu_seqlens_m=offsets,
        A_idx=gather,
        preact_out=h,
        postact_out=a,
        store_preact=True,
        concat_layout=("B",),
    )
    return h, a, z


@_up_forward.register_fake
def _up_fake(
    x,
    gather,
    reverse,
    gate_up,
    a13,
    b13,
    offsets,
    top_k,
    grad_dtype,
    need_x,
    need_weight,
    need_factors,
    weight_components,
):
    rows, width = gather.shape[0], gate_up.shape[1] // 2
    return (
        x.new_empty((rows, 2 * width)),
        x.new_empty((rows, width)),
        x.new_empty((rows, a13.shape[1])),
    )


@torch.library.custom_op(
    "mlops::lora_sonicmoe_up_bwd",
    mutates_args=(),
    schema=(
        "(Tensor grad_h, Tensor x, Tensor gather, Tensor reverse, Tensor? gate_up,"
        " Tensor? a13, Tensor? b13, Tensor? z, Tensor offsets,"
        " SymInt[] weight_shape, SymInt[] a_shape, SymInt[] b_shape, SymInt top_k,"
        " ScalarType? grad_dtype, Tensor[] weight_components)"
        " -> (Tensor?, Tensor?, Tensor?, Tensor?)"
    ),
)
def _up_backward(
    grad_h: torch.Tensor,
    x: torch.Tensor,
    gather: torch.Tensor,
    reverse: torch.Tensor,
    gate_up: torch.Tensor | None,
    a13: torch.Tensor | None,
    b13: torch.Tensor | None,
    z: torch.Tensor | None,
    offsets: torch.Tensor,
    weight_shape: list[int],
    a_shape: list[int],
    b_shape: list[int],
    top_k: int,
    grad_dtype: torch.dtype | None,
    weight_components: list[torch.Tensor],
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    """``(dx, d gate_up, da13, db13)`` from the interleaved pre-activation
    gradient: the rows' gradient through the base weight with the LoRA
    branch accumulated in place, summed onto the tokens (``gate_up``
    present); the base weight's gradient (``weight_shape`` nonempty); the
    factors' gradients (``z`` present)."""
    functional = kernels()
    dh = grad_h.contiguous()
    rows, width = dh.shape[0], x.shape[1]
    dz = None if b13 is None else _through(dh, b13, offsets)
    dx = None
    if gate_up is not None:
        if weight_components:
            expanded = fp8.input_gradient_rows(
                dh, _weight(weight_components, gate_up), offsets
            )
        else:
            expanded = torch.empty(rows, width, dtype=dh.dtype, device=dh.device)
            functional.backward._up_projection_backward_act(
                w1=gate_up.permute(1, 2, 0),
                dx_expanded=expanded,
                dh=dh,
                db1=None,
                expert_frequency_offset=offsets,
                is_glu_activation=True,
                concat_layout=True,
            )
        if a13 is not None:
            _accumulate_through(dz, a13, expanded, offsets)
        dx = torch.empty_like(x)
        functional.backward._token_broadcast_backward(
            dx_reduced=dx,
            dx_expanded=expanded,
            s_reverse_scatter_idx=reverse,
            num_activated_expert_per_token_offset=None,
            varlen_K_max=int(top_k),
            H=width,
            is_varlen_K=False,
        )
    dtype = x.dtype if grad_dtype is None else grad_dtype
    dw = None
    if weight_shape[0] > 0:
        dw = _empty(weight_shape, dtype, x.device)
        _per_expert_product(x, dh, dw.transpose(1, 2), offsets, gather, ("out",))
    da13 = db13 = None
    if z is not None:
        da13 = _empty(a_shape, dtype, x.device)
        _per_expert_product(x, dz, da13.transpose(1, 2), offsets, gather)
        db13 = _empty(b_shape, dtype, x.device)
        _per_expert_product(dh, z, db13, offsets)
    return dx, dw, da13, db13


@_up_backward.register_fake
def _up_backward_fake(
    grad_h,
    x,
    gather,
    reverse,
    gate_up,
    a13,
    b13,
    z,
    offsets,
    weight_shape,
    a_shape,
    b_shape,
    top_k,
    grad_dtype,
    weight_components,
):
    dtype = grad_dtype or x.dtype
    return (
        None if gate_up is None else torch.empty_like(x),
        None if weight_shape[0] == 0 else _empty(weight_shape, dtype, x.device),
        None if z is None else _empty(a_shape, dtype, x.device),
        None if z is None else _empty(b_shape, dtype, x.device),
    )


def _up_context(ctx, inputs, output):
    (
        x,
        gather,
        reverse,
        gate_up,
        a13,
        b13,
        offsets,
        top_k,
        grad_dtype,
        need_x,
        need_weight,
        need_factors,
        components,
    ) = inputs
    _h, a, z = output
    # x (the tokens) serves the rows' gradient's shape, the base weight's
    # gradient and da13; z only db13; a13 the branch of the rows' gradient;
    # b13 both uses of dz. Neither h nor the activation is saved here: the
    # down operator keeps h and hands back its gradient.
    ctx.save_for_backward(
        x,
        gather,
        reverse,
        gate_up if need_x else None,
        a13 if need_x else None,
        b13 if (need_x or need_factors) else None,
        z if need_factors else None,
        offsets,
        *(components if need_x else ()),
    )
    ctx.weight_shape = tuple(gate_up.shape) if need_weight else (0, 0, 0)
    ctx.a_shape, ctx.b_shape = tuple(a13.shape), tuple(b13.shape)
    ctx.top_k, ctx.grad_dtype = int(top_k), grad_dtype
    ctx.set_materialize_grads(False)  # no zeros for a and z, which take none
    ctx.mark_non_differentiable(a, z)


def _up_autograd(ctx, grad_h, _grad_a, _grad_z):
    x, gather, reverse, gate_up, a13, b13, z, offsets, *components = ctx.saved_tensors
    dx, dw, da13, db13 = _up_backward(
        grad_h,
        x,
        gather,
        reverse,
        gate_up,
        a13,
        b13,
        z,
        offsets,
        list(ctx.weight_shape),
        list(ctx.a_shape),
        list(ctx.b_shape),
        ctx.top_k,
        ctx.grad_dtype,
        list(components),
    )
    return (
        dx,
        None,
        None,
        dw,
        da13,
        db13,
        None,
        None,
        None,
        None,
        None,
        None,
        [None] * len(components),
    )


_up_forward.register_autograd(_up_autograd, setup_context=_up_context)


@torch.library.custom_op("mlops::lora_sonicmoe_down_fwd", mutates_args=())
def _down_forward(
    a: torch.Tensor,
    h: torch.Tensor,
    down: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
    offsets: torch.Tensor,
    gather: torch.Tensor,
    scatter: torch.Tensor,
    reverse: torch.Tensor,
    scores: torch.Tensor,
    residual: torch.Tensor,
    top_k: int,
    grad_dtype: torch.dtype | None,
    need_h: bool,
    need_weight: bool,
    need_factors: bool,
    weight_components: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``z = a @ a2ᵀ``, ``y = a @ downᵀ + z @ b2ᵀ`` over expert-sorted rows,
    then every token's rows summed with their ``scores`` (``[tokens * top_k]``
    in token-slot order) onto ``residual``: ``(output, z)``; ``b2`` comes
    scaled. ``a`` is ``swiglu(h)``, ``h`` the interleaved pre-activation the
    backward differentiates through (``a`` itself is not saved);
    ``gather`` and ``scatter`` serve the backward."""
    from quack.gemm_interface import gemm_add

    functional = kernels()
    z = _low(a, a2, offsets)
    c = _low(z, b2, offsets)
    if weight_components:
        y = fp8.lora_down_projection(a, _weight(weight_components, down), c, offsets)
    else:
        y = torch.empty(a.shape[0], down.shape[1], dtype=a.dtype, device=a.device)
        gemm_add(a, down.transpose(1, 2), c, out=y, cu_seqlens_m=offsets)
    combined = torch.empty_like(residual)
    functional.forward._router_forward(
        y=y,
        o=combined,
        topk_scores=scores,
        s_reverse_scatter_idx=reverse,
        num_activated_expert_per_token_offset=None,
        varlen_K_max=int(top_k),
        H=residual.shape[1],
        is_varlen_K=False,
    )
    return combined + residual, z


@_down_forward.register_fake
def _down_fake(
    a,
    h,
    down,
    a2,
    b2,
    offsets,
    gather,
    scatter,
    reverse,
    scores,
    residual,
    top_k,
    grad_dtype,
    need_h,
    need_weight,
    need_factors,
    weight_components,
):
    return torch.empty_like(residual), a.new_empty((a.shape[0], a2.shape[1]))


@torch.library.custom_op(
    "mlops::lora_sonicmoe_down_bwd",
    mutates_args=(),
    schema=(
        "(Tensor grad_output, Tensor h, Tensor down, Tensor a2, Tensor b2,"
        " Tensor? z, Tensor offsets, Tensor gather, Tensor scatter, Tensor scores,"
        " SymInt[] weight_shape, SymInt[] a_shape, SymInt[] b_shape,"
        " ScalarType? grad_dtype, Tensor[] weight_components)"
        " -> (Tensor, Tensor?, Tensor?, Tensor?, Tensor)"
    ),
)
def _down_backward(
    grad_output: torch.Tensor,
    h: torch.Tensor,
    down: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
    z: torch.Tensor | None,
    offsets: torch.Tensor,
    gather: torch.Tensor,
    scatter: torch.Tensor,
    scores: torch.Tensor,
    weight_shape: list[int],
    a_shape: list[int],
    b_shape: list[int],
    grad_dtype: torch.dtype | None,
    weight_components: list[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    """``(dh, d down, da2, db2, dscores)`` from the token cotangent: the rows'
    unscaled gradient ``r = dy @ down`` gathered through ``A_idx`` with the
    LoRA branch accumulated, then one row pass over ``r`` and ``h`` for the
    score gradient, the pre-activation gradient and (base or factors
    trainable) the scaled activation ``a'``, the operand of ``d down =
    dyᵀ a'`` and ``da2 = dzᵀ a'``. The row gradient is always formed: the
    score gradient needs it."""
    from quack.gemm_interface import gemm

    kernels()
    grad_flat = grad_output.contiguous()
    rows, width = h.shape[0], h.shape[1] // 2
    if weight_components:
        r = fp8.input_gradient_rows(
            grad_flat, _weight(weight_components, down), offsets, gather=gather
        )
    else:
        r = torch.empty(rows, width, dtype=grad_flat.dtype, device=grad_flat.device)
        gemm(grad_flat, down, out=r, cu_seqlens_m=offsets, A_idx=gather)
    dz = _through(grad_flat, b2, offsets, gather)
    _accumulate_through(dz, a2, r, offsets)
    score_rows = scores[scatter.long()].contiguous()
    ds, dh, a_prime = routed_swiglu_rows_backward(
        r, h, score_rows, scaled_activation=weight_shape[0] > 0 or z is not None
    )
    dscores = torch.empty_like(scores)
    dscores[scatter.long()] = ds.to(scores.dtype)
    dtype = h.dtype if grad_dtype is None else grad_dtype
    dw = None
    if weight_shape[0] > 0:
        dw = _empty(weight_shape, dtype, h.device)
        _per_expert_product(grad_flat, a_prime, dw, offsets, gather)
    da2 = db2 = None
    if z is not None:
        da2 = _empty(a_shape, dtype, h.device)
        _per_expert_product(dz, a_prime, da2, offsets)
        db2 = _empty(b_shape, dtype, h.device)
        _per_expert_product(
            grad_flat, z * score_rows.to(z.dtype)[:, None], db2, offsets, gather
        )
    return dh, dw, da2, db2, dscores


@_down_backward.register_fake
def _down_backward_fake(
    grad_output,
    h,
    down,
    a2,
    b2,
    z,
    offsets,
    gather,
    scatter,
    scores,
    weight_shape,
    a_shape,
    b_shape,
    grad_dtype,
    weight_components,
):
    dtype = grad_dtype or h.dtype
    return (
        torch.empty_like(h),
        None if weight_shape[0] == 0 else _empty(weight_shape, dtype, h.device),
        None if z is None else _empty(a_shape, dtype, h.device),
        None if z is None else _empty(b_shape, dtype, h.device),
        torch.empty_like(scores),
    )


def _down_context(ctx, inputs, output):
    (
        _a,
        h,
        down,
        a2,
        b2,
        offsets,
        gather,
        scatter,
        _reverse,
        scores,
        _residual,
        _top_k,
        grad_dtype,
        _need_h,
        need_weight,
        need_factors,
        components,
    ) = inputs
    _out, z = output
    # The score gradient needs the full row gradient whatever else is
    # trainable, so the base weight, the factors and h are always kept.
    ctx.save_for_backward(
        h,
        down,
        a2,
        b2,
        z if need_factors else None,
        offsets,
        gather,
        scatter,
        scores,
        *components,
    )
    ctx.weight_shape = tuple(down.shape) if need_weight else (0, 0, 0)
    ctx.a_shape, ctx.b_shape = tuple(a2.shape), tuple(b2.shape)
    ctx.grad_dtype = grad_dtype
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(z)


def _down_autograd(ctx, grad_output, _grad_z):
    h, down, a2, b2, z, offsets, gather, scatter, scores, *components = (
        ctx.saved_tensors
    )
    dh, dw, da2, db2, dscores = _down_backward(
        grad_output,
        h,
        down,
        a2,
        b2,
        z,
        offsets,
        gather,
        scatter,
        scores,
        list(ctx.weight_shape),
        list(ctx.a_shape),
        list(ctx.b_shape),
        ctx.grad_dtype,
        list(components),
    )
    return (
        None,
        dh,
        dw,
        da2,
        db2,
        None,
        None,
        None,
        None,
        dscores,
        grad_output,
        None,
        None,
        None,
        None,
        None,
        [None] * len(components),
    )


_down_forward.register_autograd(_down_autograd, setup_context=_down_context)
fp8.register_container_dispatch(
    _up_forward, _up_backward, _down_forward, _down_backward
)


@flop_formula(_up_forward)
def _up_flops(x, gather, reverse, gate_up, a13, b13, *_rest, out_val=None, **_kwargs):
    rows, packed, width = gather.shape[0], gate_up.shape[1], gate_up.shape[2]
    rank = a13.shape[1]
    return (
        2 * rows * packed * width
        + 2 * rows * rank * (width + packed)
        + 5 * rows * (packed // 2)
    )


@flop_formula(_up_backward)
def _up_backward_flops(
    grad_h,
    x,
    gather,
    reverse,
    gate_up,
    a13,
    b13,
    z,
    offsets,
    weight_shape,
    a_shape,
    b_shape,
    *_rest,
    out_val=None,
    **_kwargs,
):
    rows, packed = grad_h.shape
    width, rank = a_shape[2], a_shape[1]
    gemms = 2 * rows * packed * width
    branch = 2 * rows * rank * (packed + width)
    factors = 2 * rows * rank * (width + packed)
    return (
        gemms * ((weight_shape[0] > 0) + (gate_up is not None))
        + branch * (b13 is not None)
        + factors * (z is not None)
    )


@flop_formula(_down_forward)
def _down_flops(a, h, down, a2, b2, *_rest, out_val=None, **_kwargs):
    rows, out, inner = a.shape[0], down.shape[1], down.shape[2]
    rank = a2.shape[1]
    return 2 * rows * out * inner + 2 * rows * rank * (inner + out) + 2 * rows * out


@flop_formula(_down_backward)
def _down_backward_flops(
    grad_output,
    h,
    down,
    a2,
    b2,
    z,
    offsets,
    gather,
    scatter,
    scores,
    weight_shape,
    a_shape,
    b_shape,
    *_rest,
    out_val=None,
    **_kwargs,
):
    rows, inner = h.shape[0], h.shape[1] // 2
    out, rank = b_shape[1], a_shape[1]
    gemm = 2 * rows * out * inner
    branch = 2 * rows * rank * (out + inner)
    factors = 2 * rows * rank * (inner + out)
    return (
        gemm * (1 + (weight_shape[0] > 0))
        + branch
        + factors * (z is not None)
        + 12 * rows * inner
    )


def sonicmoe_up(
    x, gather, reverse, gate_up, a13, b13, offsets, top_k, *, grad_dtype=None
):
    """``(h, a, z)`` of the up projection with its LoRA term over the rows
    ``x[gather]``, ``h`` the differentiable output; ``gate_up`` plain or a
    container, ``a13 [E, r, H]`` and ``b13 [E, 2I, r]`` (rows interleaved,
    scaled) at the rows' dtype."""
    grad = torch.is_grad_enabled()
    return _up_forward(
        x,
        gather,
        reverse,
        gate_up,
        a13,
        b13,
        offsets,
        int(top_k),
        weight_gradient_dtype() if grad_dtype is None else grad_dtype,
        grad and x.requires_grad,
        grad and gate_up.requires_grad,
        grad and (a13.requires_grad or b13.requires_grad),
        [],
    )


def sonicmoe_down(
    a,
    h,
    down,
    a2,
    b2,
    offsets,
    gather,
    scatter,
    reverse,
    scores,
    residual,
    top_k,
    *,
    grad_dtype=None,
):
    """``(output, z)`` of the down projection with its LoRA term from the
    activation ``a = swiglu(h)``, combined onto the tokens with ``scores``
    and ``residual``; the gradient flows to ``h``. ``down`` plain or a
    container, ``a2 [E, r, I]`` and ``b2 [E, H, r]`` (scaled)."""
    grad = torch.is_grad_enabled()
    return _down_forward(
        a,
        h,
        down,
        a2,
        b2,
        offsets,
        gather,
        scatter,
        reverse,
        scores,
        residual,
        int(top_k),
        weight_gradient_dtype() if grad_dtype is None else grad_dtype,
        grad and h.requires_grad,
        grad and down.requires_grad,
        grad and (a2.requires_grad or b2.requires_grad),
        [],
    )


def expert_lora(
    hidden,
    residual,
    router_weight,
    gate_up,
    down,
    a13,
    b13,
    a2,
    b2,
    *,
    top_k,
    routing_mode,
    scale,
    compute_precision=None,
):
    """The SonicMoE form of :func:`mlops.lora.experts.expert_lora`: same
    contract, same results within bf16 rounding (FP8 rounding at
    ``"fp8_current"`` or on container weights)."""
    _precision, gate_up, down, grad_dtype = fp8.prepare(
        compute_precision, gate_up, down, hidden.dtype
    )
    flat = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    logits = F.linear(flat.to(router_weight.dtype), router_weight)
    weights, ids = routing(logits, top_k, routing_mode)
    experts = gate_up.shape[0]
    offsets, gather, scatter, reverse = metadata(ids.to(torch.int32), experts)
    counts = (offsets[1:] - offsets[:-1]).long()
    probabilities = logits.float().softmax(-1)
    probability_sum = probabilities.sum(0)
    auxiliary = experts * ((counts.float() / ids.numel()) * probabilities.mean(0)).sum()
    dtype = flat.dtype
    h, activated, _z = sonicmoe_up(
        flat,
        gather,
        reverse,
        gate_up,
        a13.to(dtype),
        (interleaved_rows(b13) * scale).to(dtype),
        offsets,
        top_k,
        grad_dtype=grad_dtype,
    )
    output, _z2 = sonicmoe_down(
        activated,
        h,
        down,
        a2.to(dtype),
        (b2 * scale).to(dtype),
        offsets,
        gather,
        scatter,
        reverse,
        weights.reshape(-1).float(),
        residual.reshape_as(flat).contiguous(),
        top_k,
        grad_dtype=grad_dtype,
    )
    return output.reshape_as(hidden), auxiliary, counts, probability_sum


__all__ = [
    "expert_lora",
    "interleaved_rows",
    "sonicmoe_down",
    "sonicmoe_up",
    "supports",
]
