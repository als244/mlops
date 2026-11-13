# Provider implementation requirements

A provider adapts a kernel package or source into MLOps' implementation registry.
It preserves the public operation while owning its private kernels and saved
state. Models call the semantic API and never import provider modules.

Use [Extending MLOps](EXTENDING.md) for the implementation workflow and source
examples. This reference defines the requirements those implementations satisfy.

## Choose the boundary

Start from the mathematical operation in [OPS.md](OPS.md):

- Use a native PyTorch graph when its computation can remain visible to capture.
- Use an opaque custom-op pair for a library or fused kernel with a raw VJP.
- Expose several registered regions when backward only needs to regenerate a
  prefix of a multi-kernel computation. `setup_context` identifies saved values;
  it cannot make a compiler replay only part of an opaque producer.

Raw kernels live in `src/mlops/kernels/` or the optional dependency. They do not
resolve implementations or silently fall back. One provider can implement
several operations, and each operation can have several exact variants.

## Registration contract

An immutable `Implementation` record identifies the operation, exact
`implementation_id`, source `provider`, priority, default determinism, support
predicate and entrypoints. Its module belongs in the explicit catalog in
`src/mlops/providers/__init__.py`; importing the catalog must tolerate missing
optional packages and must not initialize CUDA.

`apply` serves model calls. Optional `forward` and `backward` serve the explicit
API and must be present together. Both surfaces use the same raw math. The
[dispatch reference](DISPATCH.md#implementation) gives the record's fields.

For operations with a `deterministic` request, every implementation must accept
it, honor it or raise. The record's `deterministic` flag describes its default
behavior. An already ordered implementation may ignore the request.

Read `weight_gradient_dtype()` at the semantic call and pass the resolved dtype
through captured forward/backward arguments and fakes. Return weight gradients
at that dtype where the implementation controls their reduction. `None` leaves
normal behavior unchanged. A library that already rounded a gradient cannot
recover wider precision by casting it afterward. See
[precision controls](DISPATCH.md#weight-gradient-precision).

## Tensor ownership

Registries, provider objects and dispatch traces must not retain invocation
tensors. Every dependency has an explicit owner:

| Tensor | Owner and lifetime |
|---|---|
| Input or public output | Caller/module; determined by its graph |
| Saved residual | One autograd/AOT graph, from forward through matching backward |
| Reusable tables or preparation data | Caller, non-persistent module buffer or compiled artifact |
| Workspace | One invocation; dead before its forward or backward returns |

`ctx.save_for_backward` retains graph-owned references and is expected. Tables,
prepacked weights, inputs, outputs and saved residuals are not workspace.
Provider-private residuals belong in private custom-op outputs and
`setup_context`, not a universal public `extra` argument or residual object.

## Asynchronous entrypoints and metadata

Forward and backward enqueue work without device-wide synchronization or host
readback. Do not derive launch metadata with device `item()`, `tolist()`, Python
reductions or data-dependent tensor factories.

Prepare reusable offsets, tables and chunk maps from caller-owned host
configuration and pass them explicitly. For example,
[`prepare_packed_sequence_metadata`](OPS.md#prepare_packed_sequence_metadata)
provides cumulative lengths and FLA chunk indices. Its graph-visible target
keeps call-derived metadata from becoming a serialized constant.

Allocation behavior must also be repeatable. An invocation-tensor cache can
change both ownership and measured workspace on a later call. The FLA adapters
disable FLA's identity-based tensor cache (`FLA_DISABLE_TENSOR_CACHE`), including
when FLA was imported earlier, and regenerate small index tensors per call.
Compiled code and immutable scalar metadata are separate from tensor caches.

During development, enable `torch.cuda.set_sync_debug_mode("error")` and restore
the prior mode afterward. Also inspect warmed host enqueue time and a CUDA
trace: the detector is incomplete, and enqueue must return before completion.

## Opaque custom operations

Keep these pieces in the owning provider module:

1. Stateless raw forward and backward functions.
2. Custom-op targets with truthful mutation schemas.
3. Fake implementations matching real shapes, dtypes, devices, strides, storage
   offsets and aliases for supported caller layouts.
4. `setup_context` saving required inputs/outputs, and registered autograd.
5. A semantic `apply` that hides private residual outputs.
6. Forward and backward FLOP formulas delegating to the canonical estimator.

Name private targets `<operation>_<provider>_<variant>_{fwd,bwd}` so exported
graphs preserve exact implementation identity. Their residual schemas are
implementation details, documented beside their source rather than made into
another public API.

Functional training targets use `mutates_args=()` and preserve inputs, saved
state and cotangents across repeated backward. A destination-writing or storage-
donating optimization needs a separate mutable overload listing every write;
the caller must prove ownership. For example,
`moe_builtin_grouped_gemm_finish_bwd_donate_h13` is compiler-only: it may overwrite
`h13` only after its final reader. Ordinary MLOps calls do not select it.

The registry selects the implementation; PyTorch's dispatcher, fake propagation
and registered VJP still perform their usual roles. Every opaque forward must
pass `torch.library.opcheck` and fake/runtime layout comparisons.

## Explicit forward and backward

The explicit façade resolves `surface="explicit"` and invokes raw functions
under `torch.no_grad()`. It neither invokes autograd nor writes `.grad`.

```python
from mlops import explicit
from mlops.dispatch import use_implementation

with use_implementation("rms_norm", "builtin.rms_norm.triton"):
    output, residual = explicit.rms_norm.forward(x, weight, eps)
    grad_x, grad_weight = explicit.rms_norm.backward(
        grad_output, x, weight, residual
    )
```

Keep an exact override or frozen implementation manifest around both calls so
the pair cannot resolve to different variants. Apply-only providers must reject
explicit selection with a reason. See [explicit signatures](EXPLICIT_OPS.md).

## Non-differentiable state updates

Optimizer operations have no VJP. Register their functional form as apply-only,
with `forward=None` and `backward=None`. Functional, `out=` and in-place targets
share raw math; mutable schemas list every written argument and do not register
autograd. The standard optimizer façade calls the mutation form under
`torch.no_grad()`.

The [AdamW implementation](../src/mlops/providers/builtin/adamw.py) illustrates
this split: `optim/` owns the PyTorch optimizer façade, the provider owns local
tensor transitions and `kernels/` owns raw mechanics. Communication, process
groups, streams and global shard ownership belong to callers.

## Support and cost hints

`supports(...)` inspects metadata and returns `SupportResult.yes()` or
`SupportResult.no(reason)`. Check package/version, device capability, dtype,
layout, dimensions and operation-specific restrictions. Never execute a kernel
or substitute another implementation inside the predicate.

Canonical estimators under `dispatch/` define provider-independent logical work
and minimum traffic. Every custom-op FLOP formula delegates to one. Optional
`Implementation.estimate` adds physical work, traffic and invocation-local
workspace; it must not count inputs, outputs, residuals, reusable data, allocator
cache or runtime leeway as workspace. Unknown is `None`; known-zero is `0`.
Measurements remain authoritative. See [cost hints](DISPATCH.md#cost-hints).

## Implementation-specific references

| Topic | Reference |
|---|---|
| Shapes, precision, saved state and supported implementations | [Operations](OPS.md) |
| SonicMoE tuning, allocator behavior and caches | [SonicMoE tuning](sonicmoe-autotuning.md) |
| Attention and hyper-connection algorithms | [Attention](ATTENTION.md) |
| Caller-owned expert-parallel groups and communication buffers | [Expert parallelism](EXPERT_PARALLEL.md) |
| Axis, reduction, destination and layout declarations | [Compiler metadata](COMPILER_METADATA.md) |
| External binaries and their build/load lifetime | [Prepared kernels](PREPARED_KERNELS.md) |
| Required tests | [Extension validation matrix](EXTENDING.md#validation-matrix) |
