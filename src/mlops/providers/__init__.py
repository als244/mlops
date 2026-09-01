"""Explicit catalog bootstrap for package-owned operation implementations."""

from __future__ import annotations

from importlib import import_module


_REGISTERED = False

# Keep package-owned discovery explicit. Third-party entry-point discovery can
# be added later without changing the operation or implementation identities.
_CATALOG = (
    "mlops.providers.builtin.adamw",
    "mlops.providers.builtin.cross_entropy",
    "mlops.providers.native_torch.cross_entropy",
    "mlops.providers.builtin.embedding",
    "mlops.providers.native_torch.embedding",
    "mlops.providers.builtin.gelu",
    "mlops.providers.native_torch.gelu",
    "mlops.providers.builtin.layer_norm",
    "mlops.providers.native_torch.layer_norm",
    "mlops.providers.builtin.rms_norm",
    "mlops.providers.native_torch.rms_norm",
    "mlops.providers.liger.rms_norm",
    "mlops.providers.builtin.swiglu",
    "mlops.providers.native_torch.swiglu",
    "mlops.providers.builtin.rope",
    "mlops.providers.native_torch.rope",
    "mlops.providers.builtin.partial_rope",
    "mlops.providers.native_torch.partial_rope",
    "mlops.providers.builtin.flash_attention",
    "mlops.providers.native_torch.flash_attention",
    "mlops.providers.builtin.mla_attention",
    "mlops.providers.native_torch.mla_attention",
    "mlops.providers.builtin.head",
    "mlops.providers.native_torch.head",
    "mlops.providers.fla.hybrid",
    "mlops.providers.native_torch.hybrid",
    "mlops.providers.builtin.linear_mixer",
    "mlops.providers.native_torch.linear_mixer",
    "mlops.providers.builtin.dsa",
    "mlops.providers.native_torch.dsa",
    "mlops.providers.builtin.moe_composed",
    "mlops.providers.builtin.moe",
    "mlops.providers.native_torch.moe",
    "mlops.providers.scattermoe.moe",
)


def ensure_implementations_registered() -> None:
    """Import each known implementation adapter exactly once."""
    global _REGISTERED
    if _REGISTERED:
        return
    # Set the guard first because adapters may resolve imports while loading.
    _REGISTERED = True
    try:
        for module_name in _CATALOG:
            import_module(module_name)
    except Exception:
        _REGISTERED = False
        raise


__all__ = ["ensure_implementations_registered"]
