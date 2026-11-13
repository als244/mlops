"""Explicit destinations for SonicMoE parameter-gradient products.

Normal ``mlops.bootstrap`` registers these accumulating dispatcher operators
before loading saved artifacts; no compiler integration import is needed. Each
operator performs its original backward once and updates all selected parameter
gradients in their GEMM epilogues. Activation and routing outputs are unchanged.
Inactive and consumed output positions are empty tensors, permitting ordinary
Torch automatic functionalization; live unconsumed outputs retain their shapes.

Version 1 accepts contiguous, aligned factor/base destinations at the original
operator's gradient dtype. It optionally writes the up-B gradient directly in
canonical gate-then-up order and folds an explicit gradient scale into the GEMM.
Narrow destinations require explicit permission for opmath product/scale/add
followed by one final cast. Cropped/padded logical ranks, intermediate narrow
casts into wider gradients, and FP8 containers require a separate contract.
"""

from __future__ import annotations

import math

import torch

from ...dispatch.costs import flop_formula
from . import lora


class _Destinations:
    def __init__(
        self,
        destinations,
        output_indices,
        contract_version,
        gradient_scales,
        canonical_b,
        fuse_narrow_gradient_accumulation,
        shapes,
        dtype,
        device,
        *,
        up,
        weight_components,
    ):
        if contract_version != 1:
            raise ValueError("unsupported SonicMoE gradient destination contract")
        if weight_components:
            raise ValueError(
                "SonicMoE destination contract v1 requires BF16 base weights"
            )
        if canonical_b and not up:
            raise ValueError("canonical gate/up output order applies only to up-B")
        if dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("unsupported SonicMoE gradient destination dtype")
        if dtype != torch.float32 and not fuse_narrow_gradient_accumulation:
            raise ValueError(
                "narrow SonicMoE destinations require explicit fused-rounding permission"
            )
        if not output_indices or len(output_indices) != len(set(output_indices)):
            raise ValueError("destination output indices must be nonempty and unique")
        if len(destinations) != len(output_indices):
            raise ValueError("each selected output requires one destination")
        scales = (
            [1.0] * len(destinations) if gradient_scales is None else gradient_scales
        )
        if len(scales) != len(destinations) or any(
            not math.isfinite(s) for s in scales
        ):
            raise ValueError("each selected output requires one finite gradient scale")
        if canonical_b and 3 not in output_indices:
            raise ValueError("canonical up-B requires the B destination")
        self.selected = {}
        self.canonical_b = canonical_b
        occupied = []
        for index, destination, scale in zip(
            output_indices, destinations, scales, strict=True
        ):
            shape = shapes.get(index)
            if shape is None:
                raise ValueError("destination selects an inactive parameter gradient")
            if tuple(destination.shape) != tuple(shape):
                raise ValueError(
                    "destination must have the complete aligned gradient shape"
                )
            if destination.dtype != dtype or destination.device != device:
                raise ValueError(
                    "destination dtype/device must match the original gradient product"
                )
            if not destination.is_contiguous():
                raise ValueError(
                    "destination contract v1 requires contiguous canonical storage"
                )
            storage = destination.untyped_storage()._cdata
            first = destination.storage_offset() * destination.element_size()
            last = first + destination.numel() * destination.element_size()
            if any(
                storage == old and first < stop and start < last
                for old, start, stop in occupied
            ):
                raise ValueError("parameter gradient destinations must not overlap")
            occupied.append((storage, first, last))
            self.selected[index] = (destination, float(scale))

    def product(
        self,
        index,
        shape,
        dtype,
        device,
        left,
        right,
        offsets,
        *,
        gather=None,
        concat=None,
        transpose=False,
    ):
        selected = self.selected.get(index)
        if selected is None:
            return lora._parameter_product(
                None,
                index,
                shape,
                dtype,
                device,
                left,
                right,
                offsets,
                gather=gather,
                concat=concat,
                transpose=transpose,
            )
        from quack.gemm_interface import gemm_add_out

        destination, scale = selected
        output = destination.transpose(1, 2) if transpose else destination
        if index == 3 and self.canonical_b:
            if concat is not None or transpose:
                raise ValueError("unsupported canonical up-B product layout")
            concat = ("out",)
        # The reduce-add store used by gemm_add_inplace casts the product to
        # the destination dtype before adding the prior gradient. This contract
        # instead reads C into the FP32 epilogue and casts only the final sum.
        # C and D are the same caller-owned buffer; no product buffer is made.
        # Both operands need the same gate/up permutation when stored canonical.
        layout = None if concat is None else ",".join((*concat, "C"))
        gemm_add_out(
            left.T,
            right,
            output,
            out=output,
            alpha=scale,
            beta=1.0,
            cu_seqlens_k=offsets,
            A_idx=gather,
            batch_idx_permute=None,
            dynamic_scheduler=False,
            add_to_output=False,
            concat_layout=layout,
        )
        return torch.empty(0, dtype=dtype, device=device)

    def tensor_results(self, values, example):
        # Fixed Tensor returns are required by Torch auto-functionalization.
        # The graph rewrite must not observe an originally inactive result.
        return tuple(
            example.new_empty(0) if value is None or index in self.selected else value
            for index, value in enumerate(values)
        )


@torch.library.custom_op(
    "mlops::lora_sonicmoe_up_bwd_accumulate",
    mutates_args=("destinations",),
    schema="(Tensor grad_h, Tensor x, Tensor gather, Tensor reverse, Tensor? gate_up, Tensor? a13, Tensor? b13, Tensor? z, Tensor offsets, SymInt[] weight_shape, SymInt[] a_shape, SymInt[] b_shape, SymInt top_k, ScalarType? grad_dtype, Tensor[] weight_components, Tensor(a!)[] destinations, int[] output_indices, int contract_version, float[]? gradient_scales=None, bool canonical_b=False, bool fuse_narrow_gradient_accumulation=False) -> (Tensor, Tensor, Tensor, Tensor)",
)
def accumulate_up(
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
    destinations,
    output_indices,
    contract_version,
    gradient_scales=None,
    canonical_b=False,
    fuse_narrow_gradient_accumulation=False,
):
    targets = _Destinations(
        destinations,
        output_indices,
        contract_version,
        gradient_scales,
        canonical_b,
        fuse_narrow_gradient_accumulation,
        {
            1: weight_shape if weight_shape[0] > 0 else None,
            2: a_shape if z is not None else None,
            3: b_shape if z is not None else None,
        },
        x.dtype if grad_dtype is None else grad_dtype,
        x.device,
        up=True,
        weight_components=weight_components,
    )
    result = lora._up_backward_impl(
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
        gradient_outputs=targets,
    )
    return targets.tensor_results(result, x)


@accumulate_up.register_fake
def _up_fake(
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
    destinations,
    output_indices,
    contract_version,
    gradient_scales=None,
    canonical_b=False,
    fuse_narrow_gradient_accumulation=False,
):
    targets = _Destinations(
        destinations,
        output_indices,
        contract_version,
        gradient_scales,
        canonical_b,
        fuse_narrow_gradient_accumulation,
        {
            1: weight_shape if weight_shape[0] > 0 else None,
            2: a_shape if z is not None else None,
            3: b_shape if z is not None else None,
        },
        x.dtype if grad_dtype is None else grad_dtype,
        x.device,
        up=True,
        weight_components=weight_components,
    )
    result = lora._up_backward_fake(
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
    )
    return targets.tensor_results(result, x)


flop_formula(accumulate_up)(lora._up_backward_flops)


@torch.library.custom_op(
    "mlops::lora_sonicmoe_down_bwd_accumulate",
    mutates_args=("destinations",),
    schema="(Tensor grad_output, Tensor h, Tensor down, Tensor a2, Tensor b2, Tensor? z, Tensor offsets, Tensor gather, Tensor scatter, Tensor scores, SymInt[] weight_shape, SymInt[] a_shape, SymInt[] b_shape, ScalarType? grad_dtype, float? swiglu_limit, Tensor[] weight_components, Tensor(a!)[] destinations, int[] output_indices, int contract_version, float[]? gradient_scales=None, bool canonical_b=False, bool fuse_narrow_gradient_accumulation=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
def accumulate_down(
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
    swiglu_limit,
    weight_components,
    destinations,
    output_indices,
    contract_version,
    gradient_scales=None,
    canonical_b=False,
    fuse_narrow_gradient_accumulation=False,
):
    targets = _Destinations(
        destinations,
        output_indices,
        contract_version,
        gradient_scales,
        canonical_b,
        fuse_narrow_gradient_accumulation,
        {
            1: weight_shape if weight_shape[0] > 0 else None,
            2: a_shape if z is not None else None,
            3: b_shape if z is not None else None,
        },
        h.dtype if grad_dtype is None else grad_dtype,
        h.device,
        up=False,
        weight_components=weight_components,
    )
    result = lora._down_backward_impl(
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
        swiglu_limit,
        weight_components,
        gradient_outputs=targets,
    )
    return targets.tensor_results(result, h)


@accumulate_down.register_fake
def _down_fake(
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
    swiglu_limit,
    weight_components,
    destinations,
    output_indices,
    contract_version,
    gradient_scales=None,
    canonical_b=False,
    fuse_narrow_gradient_accumulation=False,
):
    targets = _Destinations(
        destinations,
        output_indices,
        contract_version,
        gradient_scales,
        canonical_b,
        fuse_narrow_gradient_accumulation,
        {
            1: weight_shape if weight_shape[0] > 0 else None,
            2: a_shape if z is not None else None,
            3: b_shape if z is not None else None,
        },
        h.dtype if grad_dtype is None else grad_dtype,
        h.device,
        up=False,
        weight_components=weight_components,
    )
    result = lora._down_backward_fake(
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
        swiglu_limit,
        weight_components,
    )
    return targets.tensor_results(result, h)


flop_formula(accumulate_down)(lora._down_backward_flops)
