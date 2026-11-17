"""Full-projection PyTorch oracle for installed token-head objectives."""

from torch.nn import functional as F

from ...dispatch.registry import Implementation, register_implementation
from ...objectives._head import evaluate, validate
from ..builtin.head_objective import _policy, _supports


def apply(
    hidden,
    head_weight,
    targets,
    *,
    objective,
    inputs=(),
    controls=(),
    lora_a=None,
    lora_b=None,
    scale=1.0,
    chunk_size=None,
):
    _policy(hidden, head_weight, targets, inputs, controls, lora_a, lora_b, chunk_size)
    rows = targets.numel()
    x = hidden.reshape(rows, hidden.shape[-1])
    logits = x @ head_weight.T
    if lora_a is not None:
        a, b = lora_a.to(hidden.dtype), lora_b.to(hidden.dtype)
        logits = logits + scale * ((x @ a.T) @ b.T)
    values = -F.cross_entropy(
        logits.float(), targets.reshape(rows).long(), reduction="none"
    )
    fields = [v.reshape(rows) for v in inputs]
    if isinstance(objective, str):
        validate(objective, inputs, controls)
        loss, _gradient = evaluate(objective, values, fields, list(controls))
    else:
        loss, _gradient = objective(values, fields, list(controls))
    return loss, values.detach().reshape_as(targets)


IMPLEMENTATION = register_implementation(
    Implementation(
        operation="head_objective",
        implementation_id="native_torch.head_objective",
        provider="native_torch",
        priority=0,
        deterministic=True,
        supports=_supports,
        apply=apply,
    )
)
