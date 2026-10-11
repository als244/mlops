# Reusable operation comparisons

Run `python -m pytest -ra tests/operations`. These checks exercise general MLOps
APIs: KDA, sparse latent attention/MLA, pooled indexing, hyper-connections,
sigmoid routing, and options on the existing SwiGLU/gated RMSNorm functions.
They compare values and gradients, eager/compiled execution, operator contracts,
runtime metadata reuse and precision.

CUDA tests require SM80+; KDA and fused normalization require `mlops[fla]`,
and sparse MLA requires `mlops[tilelang]`. Missing capabilities are explicit skips.
CPU API/precision checks remain runnable without optional dependencies.

The independently sourced reference code under `references/` is pinned and
SHA256 checked; its original license is retained. Model-specific reference code
is a test oracle only. Installed operations import no test or experimental code.
