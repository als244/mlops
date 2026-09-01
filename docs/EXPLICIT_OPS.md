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
