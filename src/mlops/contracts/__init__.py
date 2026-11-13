"""Standalone operator facts for compiler authors; no engine dependency.

Declarations describe provider semantics, not scheduling or memory policy.
Callers adapt these records to their own compiler APIs.
"""

from .chunking import (
    mhc_chunk_contracts,
    operator_chunk_contracts,
    sonicmoe_chunk_contracts,
)
from .gradients import sonicmoe_gradient_contracts
from .lifetimes import fla_value_contracts
from .preparation import operator_preparations, sonicmoe_preparation
from .types import (
    AxisFrom,
    ChunkAxis,
    ChunkContract,
    ChunkLocal,
    GradientContext,
    GradientContract,
    GradientDestination,
    GradientMapping,
    KernelPreparation,
    ValueContract,
)

__all__ = [
    "AxisFrom",
    "ChunkAxis",
    "ChunkContract",
    "ChunkLocal",
    "GradientContext",
    "GradientContract",
    "GradientDestination",
    "GradientMapping",
    "KernelPreparation",
    "ValueContract",
    "fla_value_contracts",
    "mhc_chunk_contracts",
    "operator_chunk_contracts",
    "operator_preparations",
    "sonicmoe_chunk_contracts",
    "sonicmoe_gradient_contracts",
    "sonicmoe_preparation",
]
