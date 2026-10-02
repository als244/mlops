"""Prepare ordinary tensors for the independent PyTorch reference."""

import torch
import torch.distributed as dist

from mlops.expert_parallel.reference import (
    expert_computation,
    route,
    router_logits,
    routing_probabilities,
)


def torch_reference(layer, x, dy, group, expert_ids, routing_weights):
    config = layer.config
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    named = dict(layer.named_parameters())
    local = [weight.detach().dequantize() for weight in layer.expert_parameters()]
    expert_names = ["gate_up_weight", "down_weight"]
    global_weights = []
    for weight in local:
        parts = [torch.empty_like(weight) for _ in range(world)]
        dist.all_gather(parts, weight, group=group)
        global_weights.append(torch.cat(parts).float().requires_grad_())
    names = [
        name for name in named if name not in expert_names and name != "router_weight"
    ]
    plain = {
        name: named[name].detach().dequantize().clone().requires_grad_()
        for name in names
    }
    assert all(type(t) is torch.Tensor for t in [*global_weights, *plain.values()])
    value = x.detach().clone().requires_grad_()
    shared = tuple(
        plain[name]
        for name in ("shared_gate_weight", "shared_up_weight", "shared_down_weight")
        if name in plain
    )
    probabilities = routing_weights.detach().clone().requires_grad_()
    output = expert_computation(
        value,
        expert_ids,
        probabilities,
        global_weights[0][:, 0::2],
        global_weights[0][:, 1::2],
        global_weights[1],
        shared_weights=shared,
        latent_down=plain.get("latent_down_weight"),
        latent_up=plain.get("latent_up_weight"),
    )
    grads = torch.autograd.grad(
        output, (value, probabilities, *plain.values(), *global_weights), dy
    )
    expected = dict(zip(names, grads[2 : 2 + len(names)]))
    weight_grads = grads[2 + len(names) :]
    q = config.local_experts
    for gradient in weight_grads:
        dist.all_reduce(gradient, group=group)
    expected.update(
        {
            name: g[rank * q : (rank + 1) * q]
            for name, g in zip(expert_names, weight_grads)
        }
    )
    return output.detach().cpu(), [
        grads[0].detach().cpu(),
        grads[1].detach().cpu(),
        *(expected[name].detach().cpu() for name in named if name != "router_weight"),
    ]


def relative_rms(actual, expected):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return float(
        (a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)
    )


def check_router(layer, x):
    """Check routing independently, then freeze IDs/probabilities for experts.

    A tied top-k cutoff may select different expert IDs. Validate the selected
    logit values and compare probabilities/derivatives in the implementation's ID
    order, so ties cannot change which experts the expert oracle evaluates.
    """
    config = layer.config
    ids, probabilities, *_ = layer.route(x)
    value = x.detach().clone().requires_grad_()
    weight = layer.router_weight.detach().dequantize().clone().requires_grad_()
    reference_ids, _ = route(
        value,
        weight,
        top_k=config.top_k,
        dtype=config.router_dtype,
        renormalize_topk=config.renormalize_topk,
    )
    logits = router_logits(value, weight, dtype=config.router_dtype)
    selected = logits.gather(-1, ids.long()).sort(-1).values
    expected = logits.gather(-1, reference_ids.long()).sort(-1).values
    torch.testing.assert_close(selected, expected, atol=0, rtol=0)
    assert bool((ids.sort(-1).values.diff(dim=-1) > 0).all()), "Duplicate expert IDs"
    reference_p = routing_probabilities(
        logits, ids, renormalize_topk=config.renormalize_topk
    )
    cotangent = torch.randn_like(probabilities)
    actual_grads = torch.autograd.grad(
        probabilities, (x, layer.router_weight), cotangent
    )
    reference_grads = torch.autograd.grad(reference_p, (value, weight), cotangent)
    errors = {"probabilities": relative_rms(probabilities, reference_p)}
    errors.update(
        {
            name: relative_rms(a, b)
            for name, a, b in zip(("dx", "drouter"), actual_grads, reference_grads)
        }
    )
    assert errors["probabilities"] < 1e-5, errors
    assert max(errors.values()) < 0.025, errors
    return (
        ids.detach(),
        probabilities.detach().requires_grad_(),
        {
            "status": "PASS",
            "relative_rms": errors,
            "differing_id_slots": int((ids.long() != reference_ids.long()).sum()),
            "selected_logit_values_exact": True,
        },
    )
