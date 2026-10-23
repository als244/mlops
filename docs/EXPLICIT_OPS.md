# Explicit operation entrypoints

This package exposes public, stateless, autograd-independent forward and VJP
calls. Complete signatures, shapes, saved arguments, and ordered returns are in
the [Explicit operations](OPS.md#explicit-operations) API reference.

```python
from mlops.explicit import rms_norm

output, rstd = rms_norm.forward(x, weight, eps=1e-5)
grad_x, grad_weight = rms_norm.backward(
    grad_output, x, weight, rstd
)
```

There is no context object or generic `extra` value. The caller retains every
documented backward argument and passes it explicitly. Calls run under
`torch.no_grad()`, return gradients without writing `.grad`, and do not own
accumulation or tensor lifetime.

Both explicit and semantic surfaces resolve an exact implementation. They then
share that implementation's raw stateless forward/VJP functions:

```text
semantic call -> resolver -> Implementation.apply -> custom op -> raw functions
explicit call -> resolver -> Implementation.forward/backward -> raw functions
```

The explicit package does not own raw kernels and registered custom ops do not
delegate back through this public resolver. This prevents recursion, duplicate
math, and a selected implementation silently resolving again.

Only operation-level VJPs belong here. Blocks and models intentionally remain
ordinary forward-only `nn.Module` compositions; manual block/model backward
methods would recreate the frontend this project is replacing.

## Options and attention residuals

The explicit SwiGLU functions accept the same optional `limit` as their semantic
forms. Gated RMSNorm accepts `activation="silu"` (default) or `"sigmoid"`.
Use matching options in forward and backward.

`explicit.kimi_delta_attention.forward` returns output, cumulative gates and
two provider-owned intra-chunk residuals; its backward consumes these plus the
original inputs and explicit sequence metadata. Native cuLA returns empty matrix
placeholders and recomputes them during backward. Keep the same implementation
selection for forward and backward. `explicit.sparse_latent_attention.forward`
returns output and log-sum-exp; its backward combines key and value contributions
into one gradient for the shared latent bank. These paths share the ordinary
registry and artifact bootstrap. See [the attention guide](ATTENTION.md).
The sparse latent operation also accepts matching `q_rope`/`k_rope` tails in
forward and backward. Its explicit backward then returns four gradients:
query, combined latent KV, positional query, and positional key.


## Semantic compositions without explicit entrypoints

`mhc_pre` and `mhc_combine` currently expose ordinary autograd only. Their
providers register private fake/autograd/VJP operations where needed, but do
not advertise the public `explicit` surface. Direct private-op calls are not
a supported substitute. This leaves the native preprocessing composition
visible to AOTAutograd and recomputation planning.
