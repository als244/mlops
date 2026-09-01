"""Stateless ScatterMoE expert-engine primitives for the canonical MoE ABI."""

from __future__ import annotations

from scattermoe.kernels.ops import group, group_bwd_W, scatter2scatter

from ...kernels.moe_dispatch import dispatch_backward, row_dot, scale_rows_
from ...kernels.swiglu import swiglu_packed_backward, swiglu_packed_forward


def _assignment_metadata(route_ids, order, offsets):
    order_long = order.long().contiguous()
    sorted_experts = route_ids.reshape(-1).long()[order_long].contiguous()
    expert_ends = offsets[1:].long().contiguous()
    return sorted_experts, order_long, expert_ends


def forward(h2_flat, w13_experts, w2_experts, route_ids, order, offsets, top_k):
    """Return grouped first-projection residual and unweighted expert output."""
    sorted_experts, order_long, _expert_ends = _assignment_metadata(
        route_ids, order, offsets
    )
    h13 = scatter2scatter(
        h2_flat,
        w13_experts,
        sorted_experts,
        order_long,
        int(top_k),
        x_grouped=False,
        y_grouped=True,
    )
    activated = swiglu_packed_forward(h13)
    expert_output = scatter2scatter(
        activated,
        w2_experts,
        sorted_experts,
        order_long,
        1,
        x_grouped=True,
        y_grouped=True,
    )
    return h13, expert_output


def backward(
    grad_output_flat,
    h2_flat,
    w13_experts,
    w2_experts,
    route_weights,
    route_ids,
    order,
    offsets,
    slots,
    h13,
    top_k,
    *,
    swiglu_variant,
):
    """Return expert-path VJPs and route-weight cotangents without mutation."""
    sorted_experts, order_long, expert_ends = _assignment_metadata(
        route_ids, order, offsets
    )
    grouped_input = group(h2_flat, order_long, fan_out=int(top_k))
    grouped_grad = group(grad_output_flat, order_long, fan_out=int(top_k))
    activated = swiglu_packed_forward(h13)
    grad_activated = scatter2scatter(
        grouped_grad,
        w2_experts.permute(0, 2, 1).contiguous(),
        sorted_experts,
        order_long,
        1,
        x_grouped=True,
        y_grouped=True,
    )
    grad_weight_slot = row_dot(grad_activated, activated)
    grad_route_weights = (
        grad_weight_slot[slots.reshape(-1).long()]
        .view_as(route_weights)
        .contiguous()
    )

    route_scale = route_weights.reshape(-1)[order_long].float().contiguous()
    scaled_grouped_grad = grouped_grad.clone()
    scale_rows_(scaled_grouped_grad, route_scale)
    grad_w2, _bias = group_bwd_W(
        scaled_grouped_grad,
        activated,
        expert_ends,
        w2_experts.shape[0],
        has_bias=False,
    )

    scaled_grad_activated = grad_activated.clone()
    scale_rows_(scaled_grad_activated, route_scale)
    grad_h13 = swiglu_packed_backward(
        scaled_grad_activated,
        h13,
        variant=str(swiglu_variant),
    )
    grad_w13, _bias = group_bwd_W(
        grad_h13,
        grouped_input,
        expert_ends,
        w13_experts.shape[0],
        has_bias=False,
    )
    grad_grouped_input = scatter2scatter(
        grad_h13,
        w13_experts.permute(0, 2, 1).contiguous(),
        sorted_experts,
        order_long,
        1,
        x_grouped=True,
        y_grouped=True,
    )
    grad_h2 = dispatch_backward(grad_grouped_input, slots)
    return grad_h2, grad_route_weights, grad_w13, grad_w2


__all__ = ["backward", "forward"]
