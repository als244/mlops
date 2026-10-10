"""GLM language-model building blocks with ordinary PyTorch autograd.

Dense projections and full models remain caller-owned. Accelerator dependencies
are loaded only by the operation that needs them; see docs/GLM.md.
"""

from importlib import import_module

_EXPORTS = {
    "clipped_swiglu": ("activation", "clipped_swiglu"),
    "coefficients": ("hyper_connection", "coefficients"),
    "combine": ("hyper_connection", "combine"),
    "gated_rms_norm": ("gates", "gated_rms_norm"),
    "kda_decay": ("gates", "kda_decay"),
    "kimi_delta_attention": ("kda", "kimi_delta_attention"),
    "normalize_streams": ("hyper_connection", "normalize_streams"),
    "pooled_topk": ("indexing", "pooled_topk"),
    "route": ("routing", "route"),
    "sparse_latent_attention": ("attention", "sparse_latent_attention"),
    "sparse_mla": ("mla", "sparse_mla"),
}
__all__ = [
    'clipped_swiglu',
    'coefficients',
    'combine',
    'gated_rms_norm',
    'kda_decay',
    'kimi_delta_attention',
    'normalize_streams',
    'pooled_topk',
    'route',
    'sparse_latent_attention',
    'sparse_mla',
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), symbol)
    globals()[name] = value
    return value
