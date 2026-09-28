"""Contributor-facing control API for stateless per-operation dispatch."""

from .context import (
    capture_dispatch,
    deterministic_kernels,
    deterministic_required,
    dispatch_manifest,
    set_deterministic_kernels,
    set_implementations,
    set_weight_gradient_dtype,
    use_implementation,
    use_implementations,
    weight_gradient_dtype,
    weight_gradients_at,
)
from .costs import CostHints, estimate_implementation, flop_formula, has_flop_formula
from .gradcheck import (
    GradcheckCase,
    GradcheckResult,
    gradcheck_implementation,
    gradcheck_implementations,
)
from .registry import Implementation, SupportResult, implementation_registry
from .resolution import (
    explain_implementation,
    implementation_pairs,
    resolve_implementation,
)

__all__ = [
    "Implementation",
    "CostHints",
    "GradcheckCase",
    "GradcheckResult",
    "SupportResult",
    "capture_dispatch",
    "deterministic_kernels",
    "deterministic_required",
    "dispatch_manifest",
    "explain_implementation",
    "estimate_implementation",
    "flop_formula",
    "gradcheck_implementation",
    "gradcheck_implementations",
    "has_flop_formula",
    "implementation_pairs",
    "implementation_registry",
    "resolve_implementation",
    "set_deterministic_kernels",
    "set_implementations",
    "set_weight_gradient_dtype",
    "use_implementation",
    "use_implementations",
    "weight_gradient_dtype",
    "weight_gradients_at",
]
