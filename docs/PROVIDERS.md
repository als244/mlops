# Adding an operation implementation

Directories under `src/mlops/providers/` adapt one package or source into the operation library's
exact implementation registry. A provider may implement one operation, several
operations, or several variants of one operation. Models never import provider
modules directly.

## Contents

1. [Choose the boundary](#choose-the-boundary)
2. [Registration contract](#registration-contract)
3. [Tensor ownership](#tensor-ownership)
4. [Asynchronous entrypoints and metadata](#asynchronous-entrypoints-and-metadata)
5. [Opaque custom operations](#opaque-custom-operations)
6. [Non-differentiable state-update operations](#non-differentiable-state-update-operations)
7. [Explicit forward and backward](#explicit-forward-and-backward)
8. [Support and cost hints](#support-and-cost-hints)
9. [Atomic example: Liger RMSNorm](#atomic-example-liger-rmsnorm)
10. [Composite example: ScatterMoE](#composite-example-scattermoe)
11. [Future SonicMoE compatibility](#future-sonicmoe-compatibility)
12. [Future QuACK compatibility](#future-quack-compatibility)
13. [Acceptance checklist](#acceptance-checklist)

## Choose the boundary

Start from the public mathematical operation in [OPS.md](OPS.md), not
from a third-party module class. An implementation adapter must preserve that
operation's model-facing signature and outputs.

- Use an ordinary native-Torch graph when all math is captureable and no raw
  explicit VJP is available.
- Use a registered opaque custom-op pair when a Triton/CUDA/library call must
  remain one dispatcher target or exposes a low-level forward/backward pair.
- For a multi-kernel implementation, use a graph-visible composition of
  registered regions when backward can regenerate its residuals from only a
  prefix. `setup_context` declares required values, not a partial regeneration
  algorithm; one overly broad opaque producer can only be replayed in full.
- Keep raw kernels in `src/mlops/kernels/` or inside the optional dependency.
  Kernel files never resolve implementations or silently fall back.

## Registration contract

Register one immutable record:

```python
Implementation(
    operation="rms_norm",
    implementation_id="example.rms_norm.variant",
    provider="example",
    priority=25,
    deterministic=True,
    supports=supports,
    apply=apply,
    forward=forward,
    backward=backward,
    estimate=estimate,
)
```

`implementation_id` is the exact selectable and serialized identity.
`provider` is descriptive source metadata only. `forward` and `backward` must
both be present or both absent. Add the provider module to the explicit catalog
in `src/mlops/providers/__init__.py`; optional-package imports must remain lazy.

`apply` serves ordinary PyTorch model code. `forward`/`backward` serve the
autograd-independent explicit API. Both surfaces call the same raw math.

## Tensor ownership

The registry and provider adapter are stateless: neither may retain an input,
output, gradient, residual, table, or temporary tensor.

| Tensor category | Owner | Lifetime |
|---|---|---|
| operation input | caller or module | determined by caller/graph |
| autograd residual | one PyTorch autograd/AOT graph | forward through matching backward |
| public output | caller | after return |
| reusable preparation data | caller, non-persistent module buffer, or compiled artifact | explicit reusable lifetime |
| workspace | the implementation invocation | allocated and dead within one forward or backward call |

Normal `ctx.save_for_backward` is expected. It retains graph-owned references;
it does not create provider state. Reusable data such as RoPE cosine/sine tables
must be ordinary explicit inputs. It is not workspace, even when the VJP saves
a reference to it.

Do not add a universal `extra` argument or residual object. If an opaque
implementation needs different private backward state, expose that state only
in its private custom-op schema and save it in `setup_context`.

## Asynchronous entrypoints and metadata

Forward and backward entrypoints must enqueue work without device-wide
synchronization or host readback. Never call `item()`, `tolist()`, Python
reductions, or count-driven tensor factories on device values to construct
launch metadata. Such a read makes completion invisible to an asynchronous
caller and can cause a profiler to attribute unrelated stream work to the op.

When offsets, chunk maps, routing indices, tables, or similar values are known
from host round/configuration data, prepare them once and pass caller-owned
tensors explicitly. For example, `prepare_packed_sequence_metadata` returns
the cumulative lengths and FLA chunk indices shared by every Gated-DeltaNet
layer in a packed round. Both packed causal convolution and linear attention
consume those tensors; their implementations do not build or cache them.
The helper is itself a small graph-visible, non-differentiable custom target so
FakeTensor/AOT cannot mistake call-derived metadata for a serialized constant.

Use `torch.cuda.set_sync_debug_mode("error")` as a development check, restoring
the prior mode afterward. PyTorch documents this detector as experimental and
incomplete, so also inspect warmed host-enqueue latency and a CUDA profiler
timeline. The enqueue call should return before device completion.

## Opaque custom operations

Keep the complete adapter together in the owning provider module:

1. raw stateless `forward` and `backward` functions;
2. functional `torch.library.custom_op(..., mutates_args=())` targets;
3. fake implementations with exact dtype/device/shape/stride semantics;
4. `setup_context` with ordinary `ctx.save_for_backward` references;
5. `register_autograd` for the forward target; and
6. `apply`, which calls the forward target and returns only semantic outputs.

The registry chooses an implementation; it does not replace PyTorch's
dispatcher or autograd. The custom op gives Export/AOT a stable opaque target,
the fake function makes metadata propagation possible, and `register_autograd`
defines the local VJP that AOT composes into a stage backward.

Name targets `<operation>_<provider>_<variant>_{fwd,bwd}`. Treat saved tensors
and incoming cotangents as immutable. If performance requires mutation, expose
it as a separate truthful mutable execution contract; never hide it inside the
functional training op.

## Non-differentiable state-update operations

Optimizer updates such as `adamw` are valid operation boundaries even though
they have no VJP. Keep their raw kernel in `kernels/`, registered functional,
`out=`, and mutation targets in the owning provider module, and the standard
`torch.optim.Optimizer` façade under `optim/`.

The functional target remains the compiler/reference semantic form and uses
`mutates_args=()`. Distinct-destination and in-place targets must list every
written argument in `mutates_args`; they do not register autograd. Register the
functional form as an apply-only `Implementation`, with `forward=None` and
`backward=None`. The public optimizer advertises the exact implementation ID
and calls the mutation form under `torch.no_grad()`.

The builtin mixed-dtype AdamW layout is the concrete example:

```text
optim/adamw_optimizer.py       -> standard mlops.optim.AdamW
optim/distributed_adamw.py     -> private bucket/collective/state runtime
providers/builtin/adamw.py     -> registry + custom-op/fake boundaries
kernels/adamw.py               -> allocation-free Triton mechanics
```

The provider layer owns only stateless local tensor transitions. Distributed
replication/sharding composes those transitions below the optimizer façade; it
is not a second operation-registry implementation. This keeps provider custom
ops free of ProcessGroup and stream handles while allowing a compiler to link
the same local mutation target inside a larger atomic distributed task.

## Explicit forward and backward

The public explicit façade has flat mathematical signatures:

```python
output, residual = explicit.rms_norm.forward(x, weight, eps)
grad_x, grad_weight = explicit.rms_norm.backward(
    grad_output, x, weight, residual, eps
)
```

It resolves `surface="explicit"` and calls the registered raw functions under
`torch.no_grad()`. It never invokes autograd or writes `.grad`. Use an exact
override or frozen implementation manifest around both calls so a pair cannot
resolve to different variants.

Apply-only implementations must reject explicit resolution clearly.

## Support and cost hints

`supports(...)` performs metadata-only validation and returns either
`SupportResult.yes()` or `SupportResult.no(reason)`. Check dependency version,
device architecture, dtype, layout, widths, top-k, and any backend constraint.
Never execute a kernel or silently select another implementation.

`estimate(...)` is optional. It may report:

- operation-level logical FLOPs and minimum bytes;
- implementation-specific physical FLOPs and accessed bytes; and
- invocation-local workspace bytes.

Use `None` for unknown and `0` only for known-zero. Workspace excludes model
state, inputs, outputs, autograd residuals, reusable tables/prepacked weights,
allocator cache, and runtime leeway. Profiling remains authoritative.

## Atomic example: Liger RMSNorm

[`liger/rms_norm.py`](../src/mlops/providers/liger/rms_norm.py) adapts Liger Kernel 0.8.1's low-level
RMSNorm forward/backward functions. It does not monkey patch a model or use
Liger's plain `autograd.Function` as the compiler boundary.

The adapter:

- preserves Llama cast semantics, offset zero, and `in_place=False`;
- derives launch scalars from input geometry;
- gates the exact supported CUDA dtype/layout/width range;
- reports Liger's per-invocation dweight partials as backward workspace; and
- owns functional fake/autograd custom-op targets.

This is the template for a small third-party kernel integration.

## Composite example: ScatterMoE

[`scattermoe/moe.py`](../src/mlops/providers/scattermoe/moe.py) and
[`scattermoe/engine.py`](../src/mlops/providers/scattermoe/engine.py) adapt ScatterMoE 0.3.0's low-level
scatter/group operations. They do not instantiate its parameter-owning MLP or
use `ParallelLinear.apply` as the compiler boundary.

The public MoE ABI keeps routing policy, auxiliary loss, residual/shared-expert
composition, and logical parameters stable. ScatterMoE replaces only routed
expert execution. Its assignment ordering and offsets are private custom-op
residuals. Model-owned `w13`/`w2` tensors are passed directly.

This is the template for a composite provider whose private VJP state differs
from another implementation without changing model code.

## Future SonicMoE compatibility

The canonical public MoE ABI was checked against
[SonicMoE](https://github.com/Dao-AILab/sonic-moe) at commit
`0349404acd7952592f73d180ff0c1510f6d112c2`. SonicMoE exposes both fused
router/top-k execution and execution from caller-supplied sorted routing inputs,
and privately uses expert-frequency offsets plus gather/scatter/reverse-scatter
indices.

Those values should remain private to a future `sonicmoe.moe` custom-op schema.
The logical package weights `[E,H,2I]` and `[E,I,H]` correspond to SonicMoE's
`[2I,H,E]` and `[H,I,E]` views with concatenated gate/up layout. A future
adapter must validate accepted strides and make any required packing an
explicit caller/module/artifact-owned tensor rather than hidden provider state.
The initial mlops adapter is intentionally gated to SM90 and SM100 only,
regardless of any additional architectures supported upstream. Its support
function accepts exactly `(9, 0)` and `(10, 0)` until another architecture has
independent correctness, capture, and performance evidence.
The file-by-file integration workflow is documented in
[Worked example: add SonicMoE to `moe`](EXTENDING.md#worked-example-add-sonicmoe-to-moe).

## Future QuACK compatibility

[QuACK](https://github.com/Dao-AILab/quack) illustrates a provider that can
contribute multiple exact implementations. A future adapter can register
`quack.rms_norm` for RMSNorm and `quack.cross_entropy` for the existing
standalone logits-to-loss operation. These remain independent implementation
choices even though they share a package and availability helper.

The initial adapters are deliberately limited to SM90 and SM100: their support
functions accept exactly `(9, 0)` and `(10, 0)` and reject SM120 until it has
separate validation. They should call QuACK's low-level forward/backward pairs,
disable in-place backward, and own mlops custom-op/fake/registered-autograd
boundaries. The complete two-operation workflow is documented in
[Worked example: one QuACK provider for RMSNorm and cross entropy](EXTENDING.md#worked-example-one-quack-provider-for-rmsnorm-and-cross-entropy).

## Acceptance checklist

- [ ] Exact ID and conservative support reasons are registered.
- [ ] Semantic and explicit surfaces share raw math.
- [ ] No hidden fallback or retained tensor state exists.
- [ ] Warmed forward/backward enqueue without hidden device synchronization;
      caller-derived metadata is passed explicitly.
- [ ] Every opaque forward passes `torch.library.opcheck`.
- [ ] Fake outputs match real metadata.
- [ ] Repeated non-unit backward preserves saved inputs and cotangents.
- [ ] Float64-capable paths pass registry-aware `gradcheck`; low-precision-only
      paths record an explicit skip and pass direct VJP parity.
- [ ] Forward, gradients, parameters, and optimizer state pass tensor metrics.
- [ ] Export/AOT retains the exact custom-op target.
- [ ] Runtime, invocation-local workspace, and allocator peak are measured.
- [ ] A many-step real-data canary records loss and throughput.
