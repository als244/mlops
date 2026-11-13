# Implementation selection and diagnostics

Choose exact implementations, set precision requests, and inspect their support
and cost estimates. Ordinary tensor calls are documented in [OPS.md](OPS.md).

## Contents

- [Selection and precision](#selection-and-precision)
- [Implementation-control API](#implementation-control-api)
- [Cost hints](#cost-hints)
- [Gradcheck](#gradcheck)

## Selection and precision

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
`set_weight_gradient_dtype` for the requests below. The environment can make
the `set_implementations` choice for a whole process tree:
`MLOPS_IMPLEMENTATIONS=moe=builtin.moe.grouped_gemm_composed,rms_norm=...`
(comma-separated `operation=implementation_id` pairs) is applied when `mlops`
is imported, by `apply_environment_overrides`.

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

### Determinism

`deterministic_kernels` requests ordered variants of operations that support it.
The default is off; ordered variants can cost throughput. An operation that is
already ordered ignores the request; one that cannot honor it raises.
`deterministic_required` reports the setting used to resolve a
`deterministic=None` argument. This is an operation-level request, not a guarantee
of bitwise equality across hardware, dtypes, chunk geometries or an entire model.

### Weight-gradient precision

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
rounds -- FLA's causal convolution and gated RMSNorm -- come back as that
library returns them. Autograd gives
a parameter its gradient at the parameter's dtype, so eager training rounds
the result again there. A caller that manages returned gradient tensors directly
can retain the operation's output dtype. `weight_gradient_dtype`
reports the setting.


## Implementation-control API

Contributor control lives in `mlops.dispatch`. Overrides are exact,
context-local, nestable, and frozen before compiler capture.

### `Implementation`

```text
Implementation(
    operation: str,
    implementation_id: str,
    provider: str,
    priority: int,
    deterministic: bool,
    supports: Callable[..., SupportResult],
    apply: Callable,
    forward: Callable | None = None,
    backward: Callable | None = None,
    estimate: Callable | None = None,
)
```

An immutable stateless record. `provider` names only the source package.
Selection and artifact manifests always use `implementation_id`. Explicit
forward/backward are either both present or both absent.

### `SupportResult`

`SupportResult.yes()` and `SupportResult.no(reason)` report support without
executing a kernel. Forced unsupported implementations fail with this reason.

### `implementation_registry`

`implementation_registry() -> Mapping[str, Mapping[str, Implementation]]`
returns a read-only snapshot. Registry records contain no tensors.

### `resolve_implementation`

```text
resolve_implementation(
    operation: str,
    *args,
    surface: str = "semantic",
    **kwargs,
) -> Implementation
```

Filters support, then selects highest priority with exact ID as a deterministic
tie-breaker. During `torch.compile` or export capture it resolves from tensor
metadata (device, dtype, shape, and strides) without reading tensor contents,
and the captured graph keeps the selected implementation.

### `explain_implementation`

Returns the selected/forced ID and every candidate's `SupportResult` without
running the operation.

### `use_implementation`

`use_implementation(operation, implementation_id)` temporarily forces one
exact mapping and restores an outer mapping on exit.

### `use_implementations`

`use_implementations(mapping)` temporarily forces several exact operation
mappings. Unknown operation/ID pairs fail before entering the context.

### `set_implementations`

`set_implementations(mapping)` applies the same validated mappings from that
point on in the calling context, beside earlier selections. Importing `mlops`
applies `MLOPS_IMPLEMENTATIONS`, comma-separated `operation=implementation_id`
pairs, this way, so one environment variable chooses for a whole process tree.

### `capture_dispatch`

Captures exact implementation call counts during one isolated warmup. The
resolver records automatically; semantic functions contain no trace calls.

### `dispatch_manifest`

Converts a trace into one exact operation-to-implementation mapping and rejects
an operation that exercised multiple implementations.

### `implementation_pairs`

Builds exact default-versus-native-Torch mappings for verifier sweeps.

### `CostHints`

```text
CostHints(
    logical_flops: int | None = None,
    logical_bytes_accessed: int | None = None,
    implementation_flops: int | None = None,
    implementation_bytes_accessed: int | None = None,
    workspace_bytes: int | None = None,
    notes: tuple[str, ...] = (),
)
```

`None` means undefined; zero means known-zero. Workspace is scratch whose
lifetime ends when this single forward/backward call returns. It excludes
caller/module inputs, returned outputs, autograd residuals, reusable tables,
allocator cache, and runtime leeway.

### `estimate_implementation`

Returns the operation's canonical logical hints merged with the exact
implementation's optional physical/workspace hints. Every operation with an
opaque operator has a canonical estimator under `mlops.dispatch`. Common
formulas live in `logical_costs.py`; KDA, tied-latent attention and pooled-indexing
formulas live in `attention_costs.py`.
Estimation examines metadata only and retains no tensors. Profiling remains
authoritative.

### `flop_formula`

```python
flop_formula(*operators)
```

Registers the decorated function as the FLOP count of one or more registered
custom operators -- the objects `torch.library.custom_op` returns, or their
`torch.ops` packets -- with `torch.utils.flop_counter`. The function takes the
operator's arguments, positionally or by name, plus `out_val`, and returns an
`int`; it runs on fake tensors as readily as real ones and reads shapes, dtypes
and static arguments only. Every operator in the `mlops` namespace has one,
forward and backward, delegating to its operation's canonical estimator.

### `has_flop_formula`

Reports whether a registered custom operator, or its `torch.ops` packet, has a
flop formula.

### `GradcheckCase`

Defines one semantic operation invocation and optional exact implementation
IDs for batch gradcheck.

### `GradcheckResult`

Reports `passed`, `failed`, or `skipped` plus a reason.

### `gradcheck_implementation`

Calls `torch.autograd.gradcheck` for one exact implementation. Unsupported
float64/device/layout inputs are skipped; another implementation is never used.

### `gradcheck_implementations`

Runs a sequence of `GradcheckCase` values in deterministic implementation-ID
order.

### Exact implementation registry

| Operation | Default | Other registered implementations |
|---|---|---|
| `cross_entropy` | `builtin.cross_entropy.triton` | `native_torch.cross_entropy` |
| `embedding` | `builtin.embedding.deterministic` | `native_torch.embedding` |
| `gelu` | `builtin.gelu.aten_explicit_backward` | `native_torch.gelu` |
| `rms_norm` | `builtin.rms_norm.triton` | `native_torch.rms_norm` |
| `layer_norm` | `builtin.layer_norm.triton` | `native_torch.layer_norm` |
| `swiglu` | `builtin.swiglu.triton` | `builtin.swiglu.triton_legacy`, `native_torch.swiglu` |
| `packed_swiglu` | `builtin.packed_swiglu.triton` | `builtin.packed_swiglu.triton_legacy`, `native_torch.packed_swiglu` |
| `rope` | `builtin.rope.triton_table` | `builtin.rope.triton_analytic`, `native_torch.rope` |
| `partial_rope` | `builtin.partial_rope.triton` | `native_torch.partial_rope` |
| `flash_attention` | `builtin.flash_attention.aten` | `native_torch.flash_attention` |
| `head_loss` | `builtin.head_loss.chunked` | `native_torch.head_loss` |
| `lora_head_loss` | `builtin.lora_head_loss.chunked` | `native_torch.lora_head_loss` |
| `causal_conv_silu` | `fla.causal_conv_silu` | `native_torch.causal_conv_silu` |
| `causal_conv_silu_qkv` | `fla.causal_conv_silu_qkv` for supported frozen weights | `builtin.causal_conv_silu_qkv.composed` |
| `l2_norm` | `fla.l2_norm` | `native_torch.l2_norm` |
| `gated_rms_norm` | `fla.gated_rms_norm` | `native_torch.gated_rms_norm` |
| `kimi_delta_attention` | `cula.kimi_delta_attention.sm90.recompute`, `cula.kimi_delta_attention.sm90`, `fla.kimi_delta_attention.chunk` | `fla.kimi_delta_attention.chunk.ieee` (IEEE opt-in) |
| `sparse_latent_attention` | `builtin.sparse_latent_attention.hopper` on SM90, `builtin.sparse_latent_attention.tilelang`; with positional tails, `builtin.sparse_latent_attention.rope.hopper`, `builtin.sparse_latent_attention.rope.tilelang` | none |
| `pooled_topk` | `cudnn.pooled_topk.sm90`, `builtin.pooled_topk.triton` | none |
| `mhc_pre` | `deepgemm.mhc_pre`, `tilekernels.mhc_pre.sinkhorn` on SM90/SM100 | `native_torch.mhc_pre` |
| `mhc_combine` | `tilekernels.mhc_combine` on SM90/SM100, `builtin.mhc_combine.four_stream` | `native_torch.mhc_combine` |
| `linear_attention` | `fla.linear_attention.gated_delta_rule` | `native_torch.linear_attention` |
| `linear_mixer` | `builtin.linear_mixer.packed` | `native_torch.linear_mixer.per_sequence` |
| `dsa_attention` | `builtin.dsa_attention.sparse` | `native_torch.dsa_attention` |
| `dsa_indexer_key_norm` | `builtin.dsa_indexer_key_norm.layer_norm` | `native_torch.dsa_indexer_key_norm` |
| `index_scores` | `builtin.index_scores.einsum` | `native_torch.index_scores` |
| `moe` | `sonicmoe.moe.composed` on compute capability 9.x/10.x, else `builtin.moe.grouped_gemm_composed` | `sonicmoe.moe`, `builtin.moe.grouped_gemm`, `native_torch.moe` |
| `adamw` | `builtin.adamw.triton` ([OPTIMIZERS.md](OPTIMIZERS.md)) | none |


## Cost hints

`estimate_implementation(...)` returns a `CostHints` record: the operation's
canonical logical estimate merged with the exact implementation's optional
physical one. Every field is `int | None`: `None` means unknown and `0` means
known-zero. Hints are advisory; measured runtime and allocator profiling remain
authoritative.

Canonical estimates are grouped under `src/mlops/dispatch/`:
`logical_costs.py` holds common formulas and `attention_costs.py` holds KDA,
tied-latent attention and pooled-indexing formulas. There is one estimator per operation, taking the
operation's forward arguments plus `entrypoint` and reporting logical FLOPs and
minimum tensor traffic for the forward and the backward. Logical means the
mathematical work -- two per multiply-add of a matrix product, a small constant
per element of an elementwise or normalizing pass, zero for a gather -- and
work whose extent a shape cannot show is bounded from the shapes: packed
attention charges every token at `max_seqlen`, a mixture of experts charges
every assignment as one row of dense expert work. Exact implementations overlay
physical FLOPs, physical bytes, and invocation-local workspace through their
own `estimate`.

The same arithmetic reaches PyTorch's flop counter. Every custom operator in
the `mlops` namespace, forward and backward, carries a flop formula, registered
with `flop_formula(...)` beside the operator and delegating to the operation's
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
