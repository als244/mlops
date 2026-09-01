"""Contributor-facing control API for stateless per-operation dispatch."""

from .context import (
    capture_dispatch,
    dispatch_manifest,
    use_implementation,
    use_implementations,
)
from .costs import CostHints, estimate_implementation
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
    "dispatch_manifest",
    "explain_implementation",
    "estimate_implementation",
    "gradcheck_implementation",
    "gradcheck_implementations",
    "implementation_pairs",
    "implementation_registry",
    "resolve_implementation",
    "use_implementation",
    "use_implementations",
]
