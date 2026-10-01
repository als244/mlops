# Operations package architecture

`mlops` is a standalone operations and optimizer package. It has no dependency
on model classes, architecture blocks, training loops, planners, or execution
engines.

## Contents

- [Public surfaces](#public-surfaces)
- [Optimizer surface](#optimizer-surface)
- [Execution architecture](#execution-architecture)
- [Directory ownership](#directory-ownership)
- [Implementation selection](#implementation-selection)
- [PyTorch custom operations and autograd](#pytorch-custom-operations-and-autograd)
- [Tensor ownership and workspace](#tensor-ownership-and-workspace)
- [Cost hints](#cost-hints)
- [Gradcheck](#gradcheck)
- [Extending the package](#extending-the-package)

## Public surfaces

The package has two operation surfaces.

1. Semantic functions such as `rms_norm(x, weight)` are used by normal
   forward-only PyTorch modules. They participate in ordinary autograd.
2. `explicit.<operation>.forward(...)` and
   `explicit.<operation>.backward(...)` are stateless, autograd-independent
   entrypoints for callers that already manage residual tensors.

Model and block packages consume only the semantic functions exported from
`mlops`. They do not import dispatch, provider, or kernel internals.

## Optimizer surface

`mlops.optim.AdamW` follows the standard `torch.optim.Optimizer` protocol.
It owns ordinary parameter groups and checkpointable state, while delegating
each update to the same stateless tensor entrypoints available independently:

```text
AdamW.step()
    -> adamw_
    -> registered mutation boundary
    -> allocation-free Triton kernel

compiler/reference caller
    -> functional_adamw(...)  # new state version
    -> adamw(..., out=...)    # distinct destinations
    -> adamw_(...)            # proven-legal mutation
```

The optimizer advertises a stable scalar `implementation_id`. A compiler may
use that identity to link an equivalent destination-writing entrypoint, but
the optimizer never imports or knows about a compiler or execution engine.
There is deliberately no `in_place` constructor flag: PyTorch optimizers are
logically mutating. Functional and distinct-destination behavior belongs to
the lower tensor API, where ownership is explicit.

Optimizer updates use ordinary local tensors. Gradient communication, state
sharding, and master-weight ownership belong to the caller or training engine.
The optimizer advertises `supports_parameter_sharding=True` because AdamW updates
are coordinate-wise; an engine can pass owned slices through the same local
update operation. MLOps creates no process groups, communication buffers, or
optimizer-specific streams.

## Execution architecture

```text
model/block
    -> mlops semantic operation
    -> resolve exact implementation
    -> Implementation.apply
         -> native-Torch graph, or
         -> registered custom op -> raw implementation forward/VJP

manual caller
    -> mlops.explicit.<operation>
    -> resolve exact implementation
    -> Implementation.forward / Implementation.backward
    -> same raw implementation forward/VJP
```

The registry selects an implementation. It does not replace PyTorch's
dispatcher, custom-op schemas, fake implementations, or autograd system.

## Directory ownership

| Path | Responsibility |
|---|---|
| `src/mlops/<operation>.py` | public semantic façade; no kernel branches |
| `src/mlops/explicit/<operation>.py` | public explicit forward/VJP façade |
| `src/mlops/dispatch/` | scalar metadata, exact selection, overrides, tracing, logical costs and flop formulas, gradcheck |
| `src/mlops/providers/<source>/` | exact implementation adapters; opaque custom-op/fake/autograd registration |
| `src/mlops/kernels/` | raw Torch/Triton mechanics; no selection or fallback |
| `src/mlops/optim/` | standard optimizers plus functional, `out=`, and in-place update entrypoints |
| `src/mlops/bootstrap.py` | registers custom-op targets before artifact loading |

`provider` is source metadata (`builtin`, `native_torch`, `liger`,
`scattermoe`, or `fla`). Selection always uses an exact implementation ID such
as `builtin.rms_norm.triton`; there is no provider-wide switch.

## Implementation selection

The contributor-facing control API is exported by `mlops.dispatch`:

```python
implementation_registry()
resolve_implementation(operation, *args, surface="semantic", **kwargs)
explain_implementation(operation, *args, surface="semantic", **kwargs)
use_implementation(operation, implementation_id)
use_implementations({operation: implementation_id})
capture_dispatch()
implementation_pairs(operations)
```

Default resolution filters each record's `supports` result, selects the
highest priority, and uses exact implementation ID as a deterministic
tie-breaker. An exact override either runs or fails with its rejection reason;
raw implementations never choose another implementation silently.

Capture is a two-phase operation: run one ordinary warmup under
`capture_dispatch()`, convert the trace to an exact manifest, then export under
`use_implementations(manifest)`. During compiler capture the resolver accepts
only that frozen mapping, so environment discovery cannot alter an artifact.

## PyTorch custom operations and autograd

`Implementation.apply` may be a visible native-Torch graph, one opaque
registered `torch.library.custom_op`, or a graph-visible composition of opaque
registered regions. The last form preserves one public semantic operation
while exposing natural residual-lifetime boundaries to AOT. Every opaque
differentiable forward region has:

- a truthful functional schema (`mutates_args=()` in the current package);
- a fake implementation with matching count, shape, dtype, device, and layout;
- a `register_autograd` adapter;
- `setup_context` saves containing only tensors needed by its VJP; and
- a separately registered backward custom-op target when the VJP is opaque.

Fake/meta layout is tested with model-authentic views, not only convenient
contiguous tensors. For example, a router input commonly arrives as
`nn.Linear.weight.T`. Its raw GEMM VJP is contiguous, but the provider
materializes that result in the input view's stride before returning it. The
fake VJP advertises the same layout so AOT's inverse view restores the ordinary
contiguous parameter-gradient ABI.

Normal `ctx.save_for_backward` use is expected. It retains references for one
autograd graph and does not put tensors in the implementation registry. The
registry stores only callables and immutable scalar metadata.

`setup_context` declares which inputs/outputs a region's VJP consumes; it does
not define a partial recipe for rematerializing a dropped output. A large
multi-kernel implementation should therefore remain graph-visible across a
boundary where backward may need only a prefix. The composed builtin MoE, for
example, separates routing/dispatch/`w13` from SwiGLU/`w2`/combine so AOT can
regenerate `h13` without replaying the down projection. Truly monolithic
providers may remain one region and optionally supply a provider-specific
recompute/VJP preparation variant.

The explicit surface calls the same raw provider functions under
`torch.no_grad()`. It never invokes `.backward()`, writes `.grad`, or duplicates
the implementation's mathematics.

## Tensor ownership and workspace

The ownership categories are deliberately distinct:

| Category | Owner/lifetime | Example |
|---|---|---|
| operation input | caller/module, visible in the graph | parameters, sequence metadata, RoPE cosine/sine tables |
| autograd residual | reference retained for one forward/backward graph | input `x`, `rstd`, routing metadata |
| public output | caller after the invocation | normalized output, loss |
| workspace | implementation scratch that exists only during one forward or backward call | Liger's backward dweight partials |

Workspace never includes caller-owned inputs, outputs, persistent/prepared
buffers, autograd residuals that survive the call, allocator cache, or global
runtime leeway. In particular, RoPE tables are non-persistent module buffers
passed as explicit tensor inputs; they are reusable state, not workspace.

No registry, resolver, override, trace, or implementation adapter may retain a
tensor beyond the normal returned-output/autograd-context ownership above.

Provider-prepared metadata needed by backward is an ordinary residual, not
hidden registry state. For example, semantic variable-length attention creates
its compact device offsets inside the opaque forward and returns them through
the private registered boundary so `setup_context` can save them. The public
semantic result remains only the attention output; explicit callers continue
to supply offsets directly.

## Cost hints

`estimate_implementation(...)` returns a `CostHints` record: the operation's
canonical logical estimate merged with the exact implementation's optional
physical one. Every field is `int | None`: `None` means unknown and `0` means
known-zero. Hints are advisory; measured runtime and allocator profiling remain
authoritative.

The canonical estimates live in one module,
`src/mlops/dispatch/logical_costs.py`: one estimator per operation, taking the
operation's forward arguments plus `entrypoint` and reporting logical FLOPs and
minimum tensor traffic for the forward and the backward. Logical means the
mathematical work -- two per multiply-add of a matrix product, a small constant
per element of an elementwise or normalizing pass, zero for a gather -- and
work whose extent a shape cannot show is bounded from the shapes: packed
attention charges every token at `max_seqlen`, a mixture of experts charges
every assignment as one row of dense expert work. Exact implementations overlay
physical FLOPs, physical bytes, and invocation-local workspace through their
own `estimate`.

The same arithmetic reaches PyTorch's flop counter. Every registered custom
operator, forward and backward, carries a flop formula, registered with
`flop_formula(...)` beside the operator and delegating to the operation's
canonical estimator, so `torch.utils.flop_counter.FlopCounterMode` counts an
mlops operator as it counts `aten.mm`, on fake tensors as readily as real ones.
A consumer that prices a graph by arithmetic intensity -- a partitioner
deciding what is cheap enough to recompute -- therefore sees each operator as
it is rather than as unknown. `has_flop_formula(operator)` reports the
registration, and the test suite requires one of every operator in the `mlops`
namespace.

## Gradcheck

Use the registry-aware development helper rather than allowing fallback:

```python
from mlops.dispatch import (
    GradcheckCase,
    gradcheck_implementations,
)

results = gradcheck_implementations([
    GradcheckCase(
        operation="rms_norm",
        inputs=(x_double, weight_double),
    )
])
```

The helper calls `torch.autograd.gradcheck` for each exact implementation.
Float64-capable implementations must pass; low-precision-only implementations
are reported as `skipped` with their concrete support reason instead of falling
back to native Torch. Low-precision custom kernels still require direct VJP,
repeatability, immutability, and family-parity tests.

## Extending the package

Use [EXTENDING.md](EXTENDING.md) for step-by-step workflows covering a new
implementation/provider, a new semantic operation, implementation variants,
explicit entrypoints, cost estimators, optional dependencies, and raw kernels.
The lower-level adapter invariants and acceptance checklist are in
[PROVIDERS.md](PROVIDERS.md).
