# Extending MLOps

Choose an extension based on what changes for the caller. A new kernel or vendor
usually implements an existing operation; it does not need a new public API.

| Change | Files to start with |
|---|---|
| Another implementation or variant | `providers/<source>/<operation>.py`, provider catalog |
| New mathematical operation | public façade, native implementation, exports and API reference |
| Explicit forward/VJP access | `explicit/<operation>.py` and eligible provider entrypoints |
| Raw kernel | `kernels/` plus its owning provider adapter |
| Cost estimate | `dispatch/logical_costs.py` or a focused estimator module |
| Optional dependency | provider support checks and `pyproject.toml` |
| Optimizer | `optim/`, state-update provider and raw kernel |

Paths above are relative to `src/mlops/`. Read the [provider requirements](PROVIDERS.md)
for ownership, asynchronous execution, schemas and fake layouts. This page covers
where to make changes and how to validate them.

## What MLOps considers an operation

An operation has stable arguments, results and mathematical semantics independent
of its implementation. It may be a primitive such as `rms_norm`, a fused
composition such as `head_loss`, a sequence algorithm such as `linear_attention`,
or a structured computation such as `moe`. Optimizer state transitions are also
operations, although they have no VJP.

Kernel count does not define the boundary. One operation can expose several
registered regions so a compiler can regenerate a prefix without repeating the
whole computation. Parameters still belong to the caller; an operation is not a
parameter-owning model or layer.

Add a public operation only when callers need a distinct mathematical interface.
Keep ordinary compositions outside the registry unless they need independent
selection, an explicit VJP, cost estimates or an opaque compiler boundary.

## Add an implementation for an existing operation

### 1. Match the public operation

Read its [operation reference](OPS.md): arguments, shapes, dtypes, outputs,
gradients, aliases, mutations and errors. Provider-private residuals may differ,
but `Implementation.apply` must return the documented public results.

Create `src/mlops/providers/<source>/<operation>.py`. Use an exact ID such as
`example.rms_norm.variant`; provider names describe the source and are not
selection switches. One provider can register several operations independently.

### 2. Implement the math and support check

Adapt stateless raw functions that take all tensor dependencies explicitly.
Use native PyTorch when capture can see the computation. For an opaque kernel
or library call, keep the custom-op schema, fake implementation, autograd setup
and backward target together in the provider module. See
[opaque custom operations](PROVIDERS.md#opaque-custom-operations).

`supports(..., surface=...)` must inspect metadata only and explain rejection
through `SupportResult.no(reason)`. Check dependencies, hardware, dtype, layout
and dimensions. Do not run a kernel or silently fall back. An unavailable
optional dependency must disable only its own implementation.

When the public explicit API can represent the provider's saved state, expose
both raw `forward` and `backward`. Otherwise leave both absent and reject
`surface="explicit"`; do not add a generic residual object to the public API.

### 3. Register the implementation and costs

Register the immutable record and add its module to `_CATALOG` in
`src/mlops/providers/__init__.py`:

```python
from mlops.dispatch.registry import Implementation, register_implementation

implementation = register_implementation(Implementation(
    operation="rms_norm",
    implementation_id="example.rms_norm.variant",
    provider="example",
    priority=50,
    deterministic=True,
    supports=supports,
    apply=apply,
    forward=forward,
    backward=backward,
    estimate=estimate,
))
```

The callables above are defined by the adapter. `forward`/`backward` and
`estimate` are optional. Record determinism honestly and honor operation-level
precision requests; see [registration requirements](PROVIDERS.md#registration-contract).

Every registered custom operator, including backward, needs a FLOP formula
that delegates to the operation's canonical estimator:

```python
from mlops.dispatch import logical_costs as logical
from mlops.dispatch.costs import flop_formula

@flop_formula(forward_op)
def forward_flops(x, weight, *_rest, out_val=None, **_kwargs):
    return logical.rms_norm(x, weight, entrypoint="forward").logical_flops
```

An optional `estimate` reports implementation-specific work and workspace;
it does not replace the canonical logical formula. Use `None` for unknown,
not zero. [Cost hints](DISPATCH.md#cost-hints) defines the accounting.

### 4. Validate and document the implementation

Use the validation matrix below, then update the operation's implementation
notes and [registry table](DISPATCH.md#exact-implementation-registry). Add an
optional dependency to the appropriate `pyproject.toml` extra and test discovery
without it installed. Benchmark representative runtime and workspace before
raising an implementation's default priority.

### Examples in the repository

| Example | What to learn |
|---|---|
| [Builtin RMSNorm](../src/mlops/providers/builtin/rms_norm.py) | A small opaque forward, fake implementation and VJP |
| [FLA L2 normalization](../src/mlops/providers/fla/hybrid.py) | Adapting a third-party raw forward/backward pair with private residuals |
| [Composed builtin MoE](../src/mlops/providers/builtin/moe_composed.py) | Exposing a recomputable prefix behind one semantic operation |
| [SonicMoE](../src/mlops/providers/sonicmoe/moe.py) and [its composition](../src/mlops/providers/sonicmoe/moe_composed.py) | Optional dependencies, conservative support and implementation-specific routing state |
| [AdamW](../src/mlops/providers/builtin/adamw.py) | Functional and explicitly mutable state updates without autograd |

SonicMoE preserves the public `moe` arguments while keeping its sorting maps
and pre-activations private. Its support predicate checks geometry and precision
as well as hardware; use the [MoE reference](OPS.md#moe) for supported modes and
[tuning guide](sonicmoe-autotuning.md) for kernel selection. These are working
adapters to follow, rather than proposed integrations.

## Add a new semantic operation

1. **Define the interface in [OPS.md](OPS.md).** Specify the signature, tensor
   geometry, ordered results, differentiability, aliases/mutations, first-order
   VJP, sequence-reset behavior and errors. Keep vendor launch controls private.
2. **Add the public façade.** Put common validation and one resolver call in
   `src/mlops/<operation>.py`, then export it from `src/mlops/__init__.py`.
   The façade must not import providers or contain kernel branches.
3. **Provide an independent reference.** Prefer a low-priority
   `providers/native_torch/<operation>.py` implementation with CPU/float64
   coverage. If that is not possible, document the limitation and supply an
   independent numerical oracle in tests.
4. **Choose the explicit surface.** If useful, add
   `explicit/<operation>.py`, export it from `explicit/__init__.py` and document
   it in [EXPLICIT_OPS.md](EXPLICIT_OPS.md). Name saved inputs individually;
   run under `torch.no_grad()` and return gradients without writing `.grad`.
   Semantic-only operations should state why they have no explicit surface.
5. **Add accelerated implementations** using the workflow above. A visible
   ATen composition does not need a custom op.
6. **Add a canonical cost estimator** under `dispatch/` and register it with
   `register_operation_estimator`. It takes forward arguments plus `entrypoint`
   and describes logical forward/backward work independently of the provider.
7. **Complete discovery and documentation.** Update the provider catalog,
   operation index, and applicable API namespace entries. Verify exports,
   signatures, forced support failures, autograd, Export and AOT composition.

## Compiler-facing declarations

Declare independently sliceable axes, additive outputs, gradient destinations
and layout-independent inputs in [operator metadata](COMPILER_METADATA.md).
Test these facts with ordinary PyTorch. The consuming compiler owns its adapter
and integration tests; do not import a training engine here.

For external compiled kernels, use [prepared-kernel factories](PREPARED_KERNELS.md).
The provider owns its cache formats, source checks and binary validation.

## Validation matrix

| Change | Minimum validation |
|---|---|
| Native implementation | Forward/gradient parity, support and exact overrides, gradcheck where supported, Export/AOT |
| Opaque implementation | Native checks plus fake/runtime shape, dtype, device, stride, offset and aliases; `opcheck`; target retention; repeated backward and input/cotangent immutability |
| Explicit entrypoints | Arbitrary-cotangent VJP parity, no autograd recording or `.grad` writes |
| New operation | Public export/signature documentation, independent oracle and all provider surfaces |
| Costs | Arithmetic on fake tensors, `FlopCounterMode`, unknown versus zero, no execution or retained tensors |
| Raw kernel | Owning provider tests and representative runtime/workspace measurements |
| Optional dependency | Available and absent cases |
| Dispatch controls | Nested-context restoration, exact selection and tensor-stateless diagnostics |

Use model-authentic layouts, including non-contiguous inputs. Low-precision-only
kernels need direct VJP comparisons with stated tolerances; a skipped float64
gradcheck is not numerical validation. Changes to mutable operations also need
ownership and aliasing tests. Run the [operation tests](../tests/operations/README.md) and
lint before submitting; hardware-only cases must run on supported devices.
