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

## LoRA modules

`mlops.lora` provides `LoRAConfig`, `LoRALinear`, `LoRAHead`, `apply_lora`, and
`parameter_report`. [LoRA configuration and examples](LORA.md) describe target
selection, frozen defaults, head loss, expert factors and memory accounting.

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

[OPTIMIZERS.md](OPTIMIZERS.md) is the authoritative local
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

The language-model epilogues include [ordinary head loss](OPS.md#head_loss)
and [LoRA head loss](OPS.md#lora_head_loss). The latter keeps frozen base weights
free of dense weight gradients and exposes explicit factor-gradient seeds.

## Dispatch and development APIs

`mlops.dispatch` exposes exact per-operation selection and diagnostics:

```python
implementation_registry()
resolve_implementation(operation, *args, surface="semantic", **kwargs)
explain_implementation(operation, *args, surface="semantic", **kwargs)
implementation_pairs(operations)
use_implementation(operation, implementation_id)
use_implementations({operation: implementation_id})
set_implementations({operation: implementation_id})
deterministic_kernels(enabled=True)
set_deterministic_kernels(enabled=True)
deterministic_required()
weight_gradients_at(dtype)
set_weight_gradient_dtype(dtype)
weight_gradient_dtype()
capture_dispatch()
dispatch_manifest(trace)
estimate_implementation(operation, *args, entrypoint="forward", **kwargs)
flop_formula(*operators)
has_flop_formula(operator)
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
`use_implementations` applies its overrides for one block; `set_implementations`
applies the same validated overrides for the rest of the calling context, for a
process that chooses once, and so do `set_deterministic_kernels` and
`set_weight_gradient_dtype` for the requests below.

Without an override, resolution filters unsupported candidates and selects the
highest-priority remaining implementation. Variable-length FlashAttention requires
CUDA compute capability 8.0 or newer. On an RTX 2080 Ti (7.5), ordinary
`mlops.flash_attention` resolves to the PyTorch SDPA provider,
`native_torch.flash_attention`, which supports autograd and graph capture.
An exact override selecting `builtin.flash_attention.aten` remains an error on
that GPU; remove the override to allow automatic selection. The SDPA provider
serves the model-facing API, not the separate explicit forward/VJP API.

`torch.compile` and export can resolve implementations during graph capture.
Support checks inspect tensor metadata (device, dtype, shape, and strides),
without executing the model or reading tensor contents. The captured graph
contains the selected implementation; replay does not resolve providers again.
Only catalog registration and device-only extension activation run outside
tracing. Importing mlops still does not initialize CUDA.

To explicitly fix implementation identities, optionally run an eager call under
`capture_dispatch()`, obtain its `dispatch_manifest(trace)`, and use that manifest
with `use_implementations(...)`. Explicit choices are checked for support during
capture too. A full eager-model warmup is not required for automatic selection.

`deterministic_kernels` is the same kind of context-local request, but it asks
for a property rather than an identity: kernels that reach one answer by an
order that varies run to run take their ordered variant instead, so one step
from one seed lands on the same state twice. That variant costs throughput, so
the default is off and qualification turns it on. Operations that are ordered
already ignore the request, and an operation that cannot honour it raises
rather than returning an unordered result. `deterministic_required` reports the
setting, which operations read to resolve a `deterministic=None` argument.

`weight_gradients_at` asks operations for the gradients of their weights at a
dtype -- fp32, say, for a caller that keeps gradients at fp32 over bf16
weights. An operation that sums a weight's gradient over rows (a norm's weight
and bias, an embedding table, an expert's or a router's weights) keeps that
sum at fp32 and rounds it to the weight's dtype as it returns it; asked for
another dtype it returns the sum at that one, and the chunked head sums its
chunks at it. The dtype is read when the operation is called and passed to its
forward and backward operators, so a captured graph keeps the dtype it was
captured under. `None`, the default, is each weight's own dtype and leaves
every operation as it was. Weight gradients another library computes and
rounds -- FLA's causal convolution and gated RMSNorm, Liger's RMSNorm,
ScatterMoE's experts -- come back as that library returns them. Autograd gives
a parameter its gradient at the parameter's dtype, so eager training rounds
the result again there; a caller that keeps gradients itself (ShadowSpill's
`grad_dtype`) keeps what the operation returned. `weight_gradient_dtype`
reports the setting.

Cost hints contain scalar metadata only and may be undefined. The canonical
estimate of every operation with an opaque operator lives in
`mlops.dispatch.logical_costs`, one estimator per operation. `flop_formula`
registers the decorated function as the FLOP count of one or more registered
custom operators with PyTorch's flop counter, and `has_flop_formula` reports
whether an operator has one; every operator this package registers does,
forward and backward, delegating to its operation's canonical estimator.
Gradcheck uses normal semantic calls and reports low-precision-only
implementations as unsupported instead of substituting another backend.

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

## Expert-parallel modules

`mlops.expert_parallel` contains optional QuackMoE, QuackMoELoRA, TEMoE and
TEMoELoRA modules with caller-owned groups and communication buffers.
See [expert-parallel configuration, installation and examples](EXPERT_PARALLEL.md).
