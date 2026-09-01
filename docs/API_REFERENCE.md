# mlops API reference

This is the index for the public and contributor-facing `mlops` APIs. The
package requires PyTorch 2.13 or newer.

## Contents

1. [Public tensor APIs](#public-tensor-apis)
2. [Optimizer APIs](#optimizer-apis)
3. [Operation categories](#operation-categories)
4. [Dispatch and development APIs](#dispatch-and-development-apis)
5. [Private implementation boundary](#private-implementation-boundary)
6. [Contributor documentation](#contributor-documentation)

## Public tensor APIs

| Surface | Import | Use |
|---|---|---|
| Semantic operations | `from mlops import rms_norm, moe, ...` | Ordinary forward-only PyTorch code with normal autograd |
| Preparation helpers | `from mlops import prepare_packed_sequence_metadata` | Construct graph-visible, caller-owned packed-round metadata once, outside provider entrypoints |
| Explicit operations | `from mlops.explicit import rms_norm, moe, ...` | Stateless forward/VJP calls for callers that manage residual lifetimes |

[OPS.md](OPS.md) is the authoritative reference for every semantic and
explicit signature, tensor shape, dtype, return, residual, mutation rule,
implementation ID, and constraint.

## Optimizer APIs

| Surface | Import | Use |
|---|---|---|
| Standard optimizer | `from mlops.optim import AdamW` | Ordinary PyTorch training, state dicts, and compiler-recognizable implementation selection |
| Functional update | `from mlops.optim import functional_adamw` | Compiler/reference semantics returning a new state version |
| Explicit destination | `from mlops.optim import adamw` | Write a new state version to caller-owned `out=` tensors |
| In-place update | `from mlops.optim import adamw_` | Update state after the caller has established mutation legality |
| Distinct-master functional update | `from mlops.optim import functional_master_adamw` | Return model/master/moment/step state versions |
| Distinct-master destination | `from mlops.optim import master_adamw` | Write a distinct-master version to caller-owned outputs |
| Distinct-master mutation | `from mlops.optim import master_adamw_` | Update model/master/moment/step storage in place |

[OPTIMIZERS.md](OPTIMIZERS.md) is the authoritative local/distributed
optimizer, dtype, state, and tensor-entrypoint reference.

## Operation categories

| Category | Semantic API | Explicit forward/VJP API |
|---|---|---|
| Embedding | [Embedding](OPS.md#embedding) | [Embedding VJP](OPS.md#embedding-1) |
| Normalization and activations | [Normalization and activations](OPS.md#normalization-and-activations) | [Normalization](OPS.md#normalization), [activations and rotary](OPS.md#activations-and-rotary-position) |
| Rotary position | [Rotary position operations](OPS.md#rotary-position-operations) | [Activations and rotary](OPS.md#activations-and-rotary-position) |
| Dense and latent attention | [Dense and latent attention](OPS.md#dense-and-latent-attention) | [Attention and sequence operations](OPS.md#attention-and-sequence-operations) |
| DSA | [DSA indexing and selected attention](OPS.md#dsa-indexing-and-selected-attention) | [Selected-attention VJP](OPS.md#attention-and-sequence-operations) |
| Hybrid mixers | [Hybrid linear-attention operations](OPS.md#hybrid-linear-attention-operations) | [Sequence-operation VJPs](OPS.md#attention-and-sequence-operations) |
| Mixture of experts | [Mixture of experts](OPS.md#mixture-of-experts) | [MoE VJP](OPS.md#mixture-of-experts-1) |
| Language-model losses/epilogues | [Language-model epilogues](OPS.md#language-model-epilogues) | [Loss VJPs](OPS.md#losses) |

The [semantic quick index](OPS.md#quick-index) lists every public operation,
its category, output shape, and default implementation. The
[explicit quick index](OPS.md#quick-index-1) lists every operation with a
public autograd-independent entrypoint and its returned residual state.

## Dispatch and development APIs

`mlops.dispatch` exposes exact per-operation selection and diagnostics:

```python
implementation_registry()
resolve_implementation(operation, *args, surface="semantic", **kwargs)
explain_implementation(operation, *args, surface="semantic", **kwargs)
implementation_pairs(operations)
use_implementation(operation, implementation_id)
use_implementations({operation: implementation_id})
capture_dispatch()
dispatch_manifest(trace)
estimate_implementation(operation, *args, entrypoint="forward", **kwargs)
gradcheck_implementation(operation, implementation_id, inputs, **options)
gradcheck_implementations(cases, **options)
```

The same module exports the immutable records `Implementation`,
`SupportResult`, `CostHints`, `GradcheckCase`, and `GradcheckResult`. Records
describe registration metadata, support decisions, optional scalar cost
estimates, development inputs, and gradcheck outcomes respectively; none may
retain invocation tensors in global registry or cache state.

Selection overrides are context-local and exact. Unsupported forced choices
fail with their support reason; implementations never silently fall back.
Cost hints contain scalar metadata only and may be undefined. Gradcheck uses
normal semantic calls and reports low-precision-only implementations as
unsupported instead of substituting another backend.

## Private implementation boundary

Registered targets under `torch.ops.mlops` are compiler-visible internals, not
a model-facing API. Opaque provider implementations keep their custom-op
schema, fake function, `register_autograd` adapter, and raw VJP together.
Non-differentiable preparation targets keep their raw and fake definitions in
the owning preparation module and require no VJP.

## Contributor documentation

- [Package architecture and dispatch](ARCHITECTURE.md)
- [Extending the package](EXTENDING.md)
- [Provider implementation contract](PROVIDERS.md)
- [Explicit entrypoint contract](EXPLICIT_OPS.md)
- [Raw-kernel boundary](KERNELS.md)
- [Optimizer API](OPTIMIZERS.md)
