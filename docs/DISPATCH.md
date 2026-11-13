# Implementation selection and diagnostics

Choose exact implementations, set precision requests, and inspect their support
and cost estimates. Ordinary tensor calls are documented in [OPS.md](OPS.md).

## Contents

- [Selection and precision](#selection-and-precision)
- [Implementation-control API](#implementation-control-api)
- [Cost hints](#cost-hints)
- [Gradcheck](#gradcheck)

## Selection and precision

Without an override, resolution chooses the highest-priority supported
implementation, breaking ties by exact ID. Inspect support or force a choice:

```python
from mlops import rms_norm
from mlops.dispatch import explain_implementation, use_implementation

support = explain_implementation("rms_norm", x, weight)
with use_implementation("rms_norm", "builtin.rms_norm.triton"):
    output = rms_norm(x, weight)
```

Forced unsupported choices fail with their support reason. Overrides are exact,
context-local and nestable. `use_implementations(mapping)` sets several for one
block; `set_implementations(mapping)` retains them in the calling context.
Importing MLOps applies `MLOPS_IMPLEMENTATIONS`, a comma-separated list of
`operation=implementation_id` pairs, through `apply_environment_overrides`.

Capture resolves from metadata without reading tensor contents. The captured
graph keeps its selected implementation; replay does not dispatch again. A full
eager-model warmup is unnecessary. To freeze choices explicitly, record calls
under `capture_dispatch()` and pass `dispatch_manifest(trace)` to
`use_implementations(...)`. A manifest rejects an operation that resolved to
multiple implementations.

### Determinism

`deterministic_kernels(enabled=True)` requests ordered variants for one context;
`set_deterministic_kernels(enabled=True)` sets the calling context's default.
The default is off; ordered variants can cost throughput. An operation that is
already ordered ignores the request; one that cannot honor it raises.
`deterministic_required` reports the setting used to resolve a
`deterministic=None` argument. This is an operation-level request, not a guarantee
of bitwise equality across hardware, dtypes, chunk geometries or an entire model.

### Weight-gradient precision

`weight_gradients_at(dtype)` requests a weight-gradient dtype within a context;
`set_weight_gradient_dtype(dtype)` sets the calling context's default, and
`weight_gradient_dtype()` reads it. The default `None` preserves each
implementation's usual behavior.

Supported reductions can return their FP32 sums directly instead of rounding
to a narrow weight dtype. The chunked head also accumulates chunks at the
requested dtype. Capture records the setting in forward/backward arguments;
changing the ambient setting later does not alter that graph.

This cannot recover precision already discarded by a library: FLA causal
convolution and gated RMSNorm return their library's rounded gradients. Eager
autograd also stores a parameter's `.grad` at the parameter dtype. Callers that
manage returned gradient tensors directly can retain their wider dtype.


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

`estimate_implementation(...)` combines canonical logical work with the selected
implementation's optional physical estimates. These are advisory; measured
runtime and allocator evidence remain authoritative.

| Quantity | Meaning |
|---|---|
| Logical FLOPs | Mathematical work, shared across providers; matrix multiply-add counts as two |
| Logical bytes | Minimum tensor traffic implied by the operation |
| Implementation FLOPs/bytes | Additional physical estimate supplied by the provider |
| Workspace | Scratch that dies within this forward/backward invocation |

Unknown fields are `None`; zero means known-zero. Shape-only estimates bound
value-dependent work: packed attention charges each token at `max_seqlen`, and
MoE charges each assignment as dense expert work. They do not predict the actual
routing distribution or elapsed time.

Canonical estimators live in `dispatch/logical_costs.py` and focused modules
such as `attention_costs.py`. Custom-op `flop_formula` registrations delegate to
these estimators so PyTorch's `FlopCounterMode` uses the same arithmetic on real
or fake tensors. Estimation never runs kernels or retains tensors.

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
