# Operator contracts for compilers

`mlops.contracts` publishes provider semantics as standalone immutable records.
It depends on PyTorch and MLOps only. Constructing these declarations does not
initialize CUDA or import optional kernel packages. A compiler adapter converts
these records to its own APIs; the adapter belongs to the consuming project.

MLOps owns operator geometry, reduction semantics, destination eligibility and
kernel implementations. The caller owns checkpoint boundaries, execution
regions, independent-axis selection, chunk sizes, replay, spill placement and
memory admission. Models continue to express ordinary full tensors.

```python
from mlops.contracts import (
    operator_chunk_contracts,
    sonicmoe_chunk_contracts,
    sonicmoe_gradient_contracts,
    fla_value_contracts,
    operator_preparations,
)

# Describe providers already used by the model; these do not select dispatch.
chunks = (*operator_chunk_contracts(batch_size=2), *sonicmoe_chunk_contracts())
gradients = sonicmoe_gradient_contracts()
values = fla_value_contracts()
preparations = operator_preparations()
# Pass these records to the consuming compiler's adapter.
```

## Independent axes and reductions

Each `ChunkContract` contains a dispatcher target, semantic version, one entry
per output and additive-reduction positions. Changing the version must invalidate
compiled artifacts that rely on that contract. The records describe semantics,
not measured workspace or permission to replay effects.

| Output entry | Meaning |
| --- | --- |
| `AxisFrom(input_index, trailing_dims=0)` | Inherit the logical input axis, removing trailing feature dimensions before reshaping the output. |
| `ChunkAxis(dim, before=1, after=1)` | The physical dimension is ordered as `before * logical_extent * after`. |
| `ChunkLocal(dim=None)` | Coordinates belong to one invocation. If set, `dim` scales with that invocation's logical extent. These values cannot escape as an ordinary full-axis tensor. |
| `None` | Shared output; when its index is in `reductions`, sum the per-invocation contributions. |

`operator_chunk_contracts(batch_size=1)` covers normalization, SwiGLU, rotary,
gated normalization, basic mHC and builtin LoRA routing/dispatch/grouped-linear
operators. The batch factor describes flattened layouts, not a choice of axis.
A compiler must validate the declarations against its actual graph geometry.

`sonicmoe_chunk_contracts(compute_precision="bf16")` supplies seven contracts,
including accumulating backward overloads. Sorted expert rows and routing maps
stay local. Expert counts and parameter gradients are additive. Saving local
values across a boundary would require per-invocation buffers and metadata;
these declarations do not provide that representation. A compiler must reject
such an escape or choose a different cut or retention policy.

`compute_precision="fp8_block"` supplies five contracts for frozen block-FP8
bases with BF16 activations/factors. Feature-wise activation quantization stays
within each row. FP8 components remain shared. Trainable FP8 weights,
`fp8_current`, destination fusion and expert-parallel collectives are excluded;
ordinary gradient addition remains available.

`mhc_chunk_contracts(batch_size=1)` covers fused DeepGEMM preprocessing and
TileKernels postprocessing. Preprocessing requires four BF16 streams, with
32-token and 128-hidden alignment on every invocation, including tails. Shared
projection/bias/scale gradients sum. Stream/feature splitting is not certified.

## Gradient destinations

`sonicmoe_gradient_contracts()` describes fresh-backward targets, accumulating
replacements, eligible output positions, a version and an FX resolver. The
resolver takes any object satisfying `GradientContext`; it inspects tensor
metadata in `node.meta["val"]` and never edits the graph. `operations` runs from
the final gradient toward its producer, excluding tuple extraction.

The resolver returns `GradientMapping(output_indices, keywords)` or `None`.
The caller invokes the replacement once with the original arguments plus
`destinations`, `output_indices`, `contract_version` and those keywords. The
schema declares destination mutation. Selected outputs refer to their supplied
destinations in the caller's graph; unselected live outputs retain their values.

The first contribution creates a gradient; later eligible products use the
existing destination as both GEMM C and D. Each product is computed once,
without a separate product allocation/add. Empty experts preserve prior sums.
Supported destinations are complete, aligned, contiguous and at the original
FP32/BF16/FP16 gradient dtype. One finite literal scale and canonical gate/up
ordering are supported. Narrow destinations require explicit
`fuse_narrow_gradient_accumulation` permission for one final epilogue cast.
Intermediate narrowing into a wider sum, cropped/padded ranks, unknown
transforms and FP8 base containers decline fusion. One ineligible output does
not disable eligible siblings. FP16 destination tests do not imply full-model
FP16 provider qualification.

The public route overload carries a static `retain_for_backward = True`
annotation. It recommends preserving backward-required expert IDs and weights
so replay does not repeat discrete selection. A consuming compiler defines
policy precedence and storage placement. The annotation changes no arithmetic.

## Read-only values and layouts

`fla_value_contracts()` returns nine `ValueContract` records. Each target is
read-only; only its listed input positions are independent of the physical
input layout. The declarations cover explicit convolution, L2/gated
normalization and Gated DeltaNet. L2 backward certifies only its incoming
gradient; saved output/statistics retain their original layouts.

These facts allow a compiler to prove movement or view compaction, but do not
select either. Alias/effect/storage proofs, actual workspace measurements and
memory admission remain the caller's responsibility. Recurrence windows are
not independent, and no projection, expert or attention replay is authorized.

## Prepared implementations

`sonicmoe_preparation(cache_root=None)` returns a `KernelPreparation` for the
supported BF16 Sonic path. `operator_preparations(dtype=torch.bfloat16,
cache_root=None)` returns ordinary BF16 availability-based Sonic preparation
plus Hopper sparse-forward backend/LSE selection. Other dtypes return no
preparations. Device selection happens when a factory opens, not when a record
is constructed. Neither helper changes operator dispatch.

A record contains an importable `factory` and immutable `options`. Preparation
freezes implementations already selected by dispatch/autotuning; it is separate
from chunk geometry, retention and gradient fusion. See
[prepared kernels](PREPARED_KERNELS.md) for build/export, load/bind, compatibility
checks and resource ownership. Saved execution consumes actual operator targets
and prepared manifests; it must not need to rerun graph-construction callbacks.

## Validation

Standalone tests cover the records, unsupported fusion paths, fake/runtime
schemas, destination products, layout contracts, source/target checks and owned
module lifetimes. Compiler transformations and full training acceptance belong
in the consuming project's integration tests. GPU tests run only on hardware
supported by the provider; CPU contract checks remain useful on every host.
