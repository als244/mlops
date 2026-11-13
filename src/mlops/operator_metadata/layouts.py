"""Read-only and value-layout facts for the explicit FLA adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorInputLayouts:
    """Operator that only reads its inputs and supports new strides on some inputs.

    ``inputs`` lists positional/schema argument indices. At those positions,
    replacing a tensor with equal values/shape/dtype/device but different strides,
    offsets or backing storage preserves the output values. Other inputs must
    retain their layouts. This says nothing about independent chunk axes.
    """

    target: str
    version: str
    inputs: tuple[int, ...]


def fla_input_layouts() -> tuple[OperatorInputLayouts, ...]:
    """Certify value operands without choosing movement, copies or replay.

    Other ports, including saved normalization statistics, retain their original
    layouts. These facts do not make recurrence windows independent. A compiler
    must still prove aliases/effects and profile the resulting executable.
    Construction imports no FLA kernels and initializes no device context.
    """
    input_positions = {
        "causal_conv_silu_fla_fwd": (0,),
        "causal_conv_silu_fla_bwd": (0, 1),
        "causal_conv_silu_fla_input_bwd": (0, 1),
        "l2_norm_fla_fwd": (0,),
        "l2_norm_fla_bwd": (0,),
        "gated_rms_norm_fla_fwd": (0, 1),
        "gated_rms_norm_fla_bwd": (0, 1, 2),
        "linear_attention_fla_gated_delta_rule_fwd": (0, 1, 2, 3, 4),
        "linear_attention_fla_gated_delta_rule_bwd": (0, 1, 2, 3, 4, 5),
    }
    return tuple(
        OperatorInputLayouts(
            target=f"mlops.{name}.default",
            version="fla-value-operands-v1",
            inputs=inputs,
        )
        for name, inputs in input_positions.items()
    )
