"""Experimental post-autograd specialization for unused auxiliary state.

An output-only call borrows dispatch storage through gate/up, then lets down
overwrite it on the same compute stream. The normal forward and replay retain
owned saved inputs. Selection depends on graph output use, never a layer config
flag or global grad mode. Importing this module does not install compiler hooks.
"""

import operator

import torch
from torch import Tensor
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

from .operators import _checked


def _call(x, p, ids, weights, hist, handle, spec, shared_weights=None):
    runtime = _checked(handle, spec)
    w1, w2 = (
        (weights[:4], weights[4:])
        if runtime.cfg.compute_precision != "bf16"
        else weights
    )
    return runtime.forward(
        x, p, ids, hist, w1, w2, shared_weights=shared_weights, _return_state=False
    )


@torch.library.custom_op("mlops_ep_quack_output_only_v1::forward", mutates_args=())
def _forward(
    x: Tensor,
    p: Tensor,
    ids: Tensor,
    weights: list[Tensor],
    hist: Tensor,
    handle: int,
    spec: str,
) -> tuple[Tensor, list[Tensor]]:
    return _call(x, p, ids, weights, hist, handle, spec)


@_forward.register_fake
def _fake(x, p, ids, weights, hist, handle, spec):
    return torch.empty_like(x), []


@torch.library.custom_op(
    "mlops_ep_quack_output_only_v1::shared_forward", mutates_args=()
)
def _shared_forward(
    x: Tensor,
    p: Tensor,
    ids: Tensor,
    weights: list[Tensor],
    hist: Tensor,
    shared_weights: list[Tensor],
    handle: int,
    spec: str,
) -> tuple[Tensor, list[Tensor]]:
    return _call(x, p, ids, weights, hist, handle, spec, shared_weights)


@_shared_forward.register_fake
def _shared_fake(x, p, ids, weights, hist, shared_weights, handle, spec):
    return torch.empty_like(x), []


_EFFECT_HANDLES = [
    _register_effectful_op(op, _EffectType.ORDERED)
    for op in (_forward, _shared_forward)
]
REPLACEMENTS = {
    getattr(getattr(torch.ops, namespace), name).default: replacement
    for namespace, replacement in (
        ("mlops_ep_quack_v1", torch.ops.mlops_ep_quack_output_only_v1.forward.default),
        (
            "mlops_ep_quack_shared_v1",
            torch.ops.mlops_ep_quack_output_only_v1.shared_forward.default,
        ),
    )
    for name in ("forward", "bf16_forward", "fp8_forward")
}


def specialize_unused_state(graph_or_module, replacements=None):
    """Replace only calls whose entire auxiliary state has no consumers.

    Run after autograd partitioning. The replacement must preserve the ordinary
    output and effects and return an empty state list with the same schema.
    Unknown/whole-tuple consumers conservatively keep the original operation.
    Returns the number of replaced calls. No compiler-wide setting is changed.
    """
    graph = getattr(graph_or_module, "graph", graph_or_module)
    replacements = REPLACEMENTS if replacements is None else replacements
    changed = 0
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        wrapped = node.target is torch.ops.higher_order.with_effects
        original = node.args[1] if wrapped else node.target
        if original not in replacements:
            continue
        state_index = 2 if wrapped else 1
        dead = []
        safe = True
        for user in node.users:
            if (
                user.op != "call_function"
                or user.target is not operator.getitem
                or len(user.args) != 2
                or type(user.args[1]) is not int
                or not 0 <= user.args[1] <= state_index
            ):
                safe = False
                break
            if user.args[1] == state_index:
                if user.users:
                    safe = False
                    break
                dead.append(user)
        if not safe:
            continue
        for user in dead:
            graph.erase_node(user)
        replacement = replacements[original]
        if wrapped:
            node.args = (node.args[0], replacement, *node.args[2:])
        else:
            node.target = replacement
        for key in ("val", "example_value", "tensor_meta"):
            value = node.meta.get(key)
            if type(value) in (tuple, list) and len(value) == state_index + 1:
                updated = list(value)
                updated[state_index] = []
                node.meta[key] = type(value)(updated)
        changed += 1
    graph.lint()
    if hasattr(graph_or_module, "recompile"):
        graph_or_module.recompile()
    return changed
