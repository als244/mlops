"""Read-only and value-layout facts for the explicit FLA adapters."""

from .types import ValueContract


def fla_value_contracts() -> tuple[ValueContract, ...]:
    """Certify value operands without choosing movement, copies or replay.

    Other ports, including saved normalization statistics, retain their original
    layouts. These facts do not make recurrence windows independent. A compiler
    must still prove aliases/effects and profile the resulting executable.
    Construction imports no FLA kernels and initializes no device context.
    """
    ports = {
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
        ValueContract(f"mlops.{name}.default", "fla-value-operands-v1", inputs)
        for name, inputs in ports.items()
    )
