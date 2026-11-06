# ShadowSpill integration

`mlops.integrations.shadowspill` supplies explicit provider contracts for
ShadowSpill's generic build controls. ShadowSpill remains an optional caller
dependency. Importing mlops or this module does not import ShadowSpill, initialize
CUDA, select a model, or install a process-wide policy.

```python
from mlops.integrations.shadowspill import (
    sonicmoe_chunk_rules,
    sonicmoe_gradient_destination_rules,
)
from shadowspill import build_step

program = build_step(
    model,
    objective=objective,
    optimizer=optimizer,
    example_args=example_args,
    gradient_destination_rules=sonicmoe_gradient_destination_rules(),
    chunking=chunking_policy,
    graph_pair_policy=graph_pair_policy,
)
```

The model must already use the SonicMoE LoRA provider. The destination factory
registers accumulating dispatcher overloads and returns immutable build rules;
it does not change provider dispatch. A build containing no matching operators
simply does not apply them. Pass the same rules in shared `build_options` when
searching chunk sizes and retention candidates. Each resulting task still needs
normal finalization, measurement and physical memory admission.

## Gradient storage and precision

The first parameter contribution allocates its gradient. Later eligible uses
across chunks, execution/checkpoint regions and microbatches use the supplied
destination. A single backward call handles all selected parameter outputs,
using one GEMM per product with `C` and `D` referring to that destination. It
does not create a separate product tensor and then add it. Empty experts preserve
the prior contents. Activation and routing gradients and other unconsumed outputs
retain their ordinary contracts.

Version 1 supports complete aligned contiguous FP32, BF16 or FP16 destinations
at the original gradient dtype, one finite literal scale, and canonical up-B
gate/up ordering. Narrow accumulation obeys ShadowSpill's explicit
`fuse_narrow_gradient_accumulation` permission: products, scale and prior sum
use the opmath epilogue with one final cast. With that permission disabled,
narrow destinations keep ordinary accumulation. Intermediate narrowing into a
wider final gradient, cropped/padded logical ranks, unknown transformations and
FP8 base containers also keep the ordinary path; casts are never removed merely
to enable fusion. One unsupported output does not prevent another eligible
output of the same backward from using its destination.

The dtype list describes the destination contract, not an expanded guarantee
about every Sonic kernel or end-to-end training dtype. Actual BF16-factor
training and two-rank data parallelism have dedicated integration coverage;
FP16 epilogue coverage alone is not FP16 model qualification.

## Chunk axes and state

`sonicmoe_chunk_rules()` returns seven `OperatorRule` objects for routing
metadata and the fresh/accumulating up/down operations. Add these to the rules
for other operators in each selected `AxisChunks` region:

```python
rules = (*other_operator_rules, *sonicmoe_chunk_rules())
# Use rules=rules in the AxisChunks values returned by chunking_policy.
```

The input domain and axis come from the caller's chunk policy. Sonic's sorted
rows and routing maps are chunk-local and must not escape a chunk; expert counts
and parameter gradients are additive reductions. An invalid saved-state cut
that exposes those local values must be changed or refused by the builder.
These rules do not select checkpoint boundaries, retention, replay, chunk sizes
or spill residency. They do not certify expert-parallel collectives, FP8 layouts
or arbitrary provider effects as independent across an axis.

## Saved execution and validation

Mapper callbacks run during graph construction. Saved Programs carry the actual
replacement overload, its arguments/version and implementation dependency;
executors need the provider implementation but do not call these factories or
mappers. Normal graph compatibility, task profiling and PressureFit admission
still apply. Full/tail sizes are distinct measured tasks where their contracts
differ. This adapter does not bundle external Triton/CuTe binaries or guarantee
that first invocation performs no external JIT compilation.

## Preparing Sonic kernels

The separate preparation adapter packages the selected BF16 Sonic/Quack
implementations with a saved Program. It is an explicit build option:

```python
from shadowspill import OperatorPreparation

prepared_sonic = OperatorPreparation("mlops.preparation.sonic:factory")
# Add operator_preparation=(prepared_sonic,) to build_forward,
# build_forward_backward, build_step, or shared search build_options.
```

Constructing this description does not initialize CUDA. The builder opens its
factory before graph compilation, then captures selected external kernels in
each task's existing finalization call. Autotuner benchmark candidates are
excluded; warm winners, generated epilogues, tails and accumulation forms are
included. The task is not executed an additional time to collect binaries.

The adapter packages Quack objects, external Triton CUBINs, the native launcher
extension and the occupancy helper. Saved profiling and execution bind these
assets without compiling or tuning. Loading validates implementation sources,
GPU target, CPU/host ABI, runtime contracts and binary digests. Missing or
incompatible implementations refuse; eager caches cannot supply a fallback.
The optional `cache_root` option controls build intermediates only. Restoring
archived binaries does not require that directory to exist.

Profiling loads before task measurements. Execution installs the engine's
allocator first, reserves the profiled external footprint, then loads selected
modules before Plan layout admission and the final physical-budget check.
Prepared libraries are shared across compatible task bindings and explicitly
released after streams drain. Binding activation does not load these modules.
The native Python launcher extension has separately validated process lifetime.
Prebuilt PyTorch/cuBLAS libraries may still initialize on first use. Their
external footprint remains reserved by profiling/admission; preparation does
not eliminate every source of first-invocation latency.

Preparation is independent of destination rules, chunk axes/sizes, checkpoint
cuts and save/recompute policies. Each search candidate still builds, profiles
and plans its own selected implementations. No precision settings, routing
choices, retention decisions or gradient arithmetic change.

The initial adapter covers the BF16 Sonic path. FP8, expert parallelism and
unrelated attention providers require their own preparation contracts. A small
real BF16 saved-Plan fixture has independent gradient/checkpoint coverage;
prepared full-model, other-dtype and distributed qualification remain separate.
The provider mechanisms and lifecycle limits are described in
[prepared kernels](PREPARED_KERNELS.md).

CPU tests in `test_shadowspill_sonicmoe.py` exercise actual custom-op capture,
grouped outputs, dtype permissions, canonical layouts and conservative fallbacks.
`test_sonicmoe_gradient_destinations.py` covers schemas, functionalization and
ownership. `test_sonicmoe_gradient_destinations_gpu.py` checks real GEMM products,
same-buffer epilogues and empty experts against independent CPU FP64 products.
The optional engine tests require a ShadowSpill version exporting the destination
and arbitrary-axis chunking APIs.
