"""Operator metadata for compiler authors; no execution-engine dependency.

Chunking describes output axes and sums. Gradients describes supported writes
into existing buffers. Layouts describes which inputs can be copied or strided.
Preparation identifies factories for saving and loading selected kernels.
Callers adapt these records to their own compiler APIs and choose all policies.
"""

from .chunking import (
    AxisFromInput,
    ChunkLocalOutput,
    FactoredAxis,
    OperatorChunking,
    mhc_chunking,
    operator_chunking,
    sonicmoe_chunking,
)
from .gradients import (
    GradientAccumulation,
    GradientAccumulationContext,
    GradientAccumulationSelection,
    GradientDestination,
    sonicmoe_gradient_accumulation,
)
from .layouts import OperatorInputLayouts, fla_input_layouts
from .preparation import KernelPreparation, operator_preparations, sonicmoe_preparation

__all__ = [
    "AxisFromInput",
    "ChunkLocalOutput",
    "FactoredAxis",
    "GradientAccumulation",
    "GradientAccumulationContext",
    "GradientAccumulationSelection",
    "GradientDestination",
    "KernelPreparation",
    "OperatorChunking",
    "OperatorInputLayouts",
    "fla_input_layouts",
    "mhc_chunking",
    "operator_chunking",
    "operator_preparations",
    "sonicmoe_chunking",
    "sonicmoe_gradient_accumulation",
    "sonicmoe_preparation",
]
