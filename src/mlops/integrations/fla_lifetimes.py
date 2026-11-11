"""Value/layout contracts for the explicit FLA adapters, without policy choices."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shadowspill.pytorch import (
        ExecutionLifetimes,
        LifetimeOperation,
        LifetimeRule,
        LifetimeSelection,
    )


def fla_execution_lifetimes(
    *,
    rules: tuple[LifetimeRule, ...] = (),
    select: Callable[[LifetimeOperation], LifetimeSelection | None] | None = None,
) -> ExecutionLifetimes:
    """Certify FLA value operands for caller-selected movement/compaction.

    This supplies readonly and input-layout contracts for the current explicit
    convolution, L2/gated normalization and Gated DeltaNet adapters. It selects
    no operations by default: callers provide ordinary lifetime rules or a
    selector. It does not permit chunking a sequence recurrence or replaying
    expensive work. Other providers, statistics/metadata and weights retain
    their ordinary contracts. In particular, L2 backward's output/statistics
    inputs are not certified for relayout because its kernel consumes them
    directly; only its incoming gradient is normalized to contiguous storage.

    Construction imports no FLA kernels and initializes no CUDA context.
    ShadowSpill remains optional until this factory is called. Changed logical
    layouts still require the engine's alias/effect/storage proofs, fresh
    compilation/profiling and physical memory admission.
    """
    from shadowspill.pytorch import (
        ExecutionLifetimes,
        LayoutIndependentOperator,
        ReadonlyOperator,
    )

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
    version = "fla-value-operands-v1"
    return ExecutionLifetimes(
        rules=rules,
        select=select,
        readonly=tuple(
            ReadonlyOperator(f"mlops.{name}.default", version) for name in ports
        ),
        layout=tuple(
            LayoutIndependentOperator(f"mlops.{name}.default", version, inputs)
            for name, inputs in ports.items()
        ),
    )
