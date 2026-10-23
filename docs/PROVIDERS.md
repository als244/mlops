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
9. [Atomic example: FLA L2 norm](#atomic-example-fla-l2-norm)
10. [Composite example: the composed builtin MoE](#composite-example-the-composed-builtin-moe)
11. [SonicMoE](#sonicmoe)
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

`deterministic` describes what the implementation does when nothing is asked
of it. Some operations also take a determinism *request*: they pass
`deterministic=` to `apply`, and every implementation of such an operation
must accept the keyword. An implementation that is ordered already records
`deterministic=True` and ignores the request; one that can be ordered on
demand records what its default does and honours the request; one that cannot
honour it raises rather than returning an unordered result. See
[`flash_attention`](OPS.md#flash_attention) for the worked case and
[`deterministic_kernels`](API_REFERENCE.md#dispatch-and-development-apis) for
the ambient form.

An implementation whose backward returns a weight's gradient reads
[`weight_gradient_dtype()`](API_REFERENCE.md#dispatch-and-development-apis) in
`apply`, passes it to its forward and backward operators -- so a captured
forward records the dtype its backward uses, and fake implementations report
it -- and returns the gradient at it, unrounded where its kernel sums at a
wider precision. `None` must leave the implementation exactly as it was. One
whose weight gradient another library computes and rounds returns it as that
library does. See [`rms_norm`](OPS.md#rms_norm) for the worked case.

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

An operation must also allocate the same way on every call: a planner
profiles a task once and replays it, and reclaims what a closed plan's tasks
allocated. FLA memoizes its own sequence-index helpers (`fla.utils.tensor_cache`,
such as the chunk offsets its delta-rule kernels derive from the cumulative
lengths) by argument identity and keeps the tensors it computed, so a result
cached under one plan and found under the next changes what a task allocates.
The FLA provider package turns that cache off when its adapters load
(`FLA_DISABLE_TENSOR_CACHE`, for FLA imported before or after them); FLA then
recomputes a few small index tensors per call.

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
providers/builtin/adamw.py     -> registry + custom-op/fake boundaries
kernels/adamw.py               -> allocation-free Triton mechanics
```

The provider layer owns stateless local tensor transitions. A training engine
can place the same transition inside a larger task containing its own gradient
communication and shard ownership. Process groups and communication streams do
not appear in the optimizer or its custom-op interface.

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

- implementation-specific physical FLOPs and accessed bytes; and
- invocation-local workspace bytes.

Operation-level logical FLOPs and minimum bytes are not the adapter's to
report: they come from the operation's canonical estimator under
`src/mlops/dispatch/`. Common formulas live in `logical_costs.py`; additional
operation families can use focused modules such as `attention_costs.py`.
The adapter gives every custom operator it registers, forward and backward,
a flop formula that
delegates to that estimator, with `flop_formula` from `mlops.dispatch.costs`,
so PyTorch's flop counter prices the operator as it prices an ATen matrix
product.

Use `None` for unknown and `0` only for known-zero. Workspace excludes model
state, inputs, outputs, autograd residuals, reusable tables/prepacked weights,
allocator cache, and runtime leeway. Profiling remains authoritative.

## Atomic example: FLA L2 norm

The `l2_norm` adapter in [`fla/hybrid.py`](../src/mlops/providers/fla/hybrid.py)
wraps flash-linear-attention's low-level L2-norm forward and backward
functions under `torch.no_grad()`. It does not use FLA's own
`autograd.Function` as the compiler boundary.

The adapter:

- gates on CUDA tensors through the provider's shared `_cuda_support` check;
- owns a functional forward custom-op target that returns the normalized
  output and the `rstd` residual, with a fake that reproduces both;
- owns a functional backward custom-op target and registers autograd that
  saves exactly the residuals the backward reads and marks `rstd`
  non-differentiable; and
- gives both custom operators flop formulas that delegate to the L2-norm
  canonical estimate.

This is the template for a small third-party kernel integration.

## Composite example: the composed builtin MoE

[`builtin/moe_composed.py`](../src/mlops/providers/builtin/moe_composed.py)
splits the routed MoE into two opaque regions -- routing and the `w13`
projection, then activation, the `w2` projection and the combine -- behind
the one public `moe` contract, so AOTAutograd can rematerialize the first
without replaying the second.

The public MoE ABI keeps routing policy, auxiliary loss, residual/shared-expert
composition, and logical parameters stable. The regions' assignment ordering,
offsets and the `h13` pre-activation are private custom-op residuals.
Model-owned `w13`/`w2` tensors are passed directly.

This is the template for a composite provider whose private VJP state differs
from another implementation without changing model code.

## SonicMoE

[`sonicmoe/moe.py`](../src/mlops/providers/sonicmoe/moe.py) and
[`sonicmoe/moe_composed.py`](../src/mlops/providers/sonicmoe/moe_composed.py)
implement `moe` on [SonicMoE](https://github.com/Dao-AILab/sonic-moe)'s fused
kernels (commit `7396f3e`, on Quack `35266c3`): the routing metadata, the
gathered up projection with SwiGLU, the down projection with the
score-weighted combine, and the fused backward kernels. Routing, the
auxiliary loss and the probability sums are the package's own
(`kernels/moe_router.py`, `providers/moe_common.py`), so every routing mode
behaves exactly as with the builtin.

A `swiglu_limit` (GLM's clamped SwiGLU, `moe`'s option) runs in the same
kernels: the gated forward and backward GEMMs use the package's own Quack
epilogues (`expert_parallel/quack/experts/swiglu_limit.py`, bf16 and FP8,
with and without a LoRA term entering before the activation), which take the
limit as a run-time scalar operand, and the row kernels
(`expert_parallel/quack/kernels/swiglu_rows.py`) take it as a run-time
argument, so one compiled kernel serves every limit.

The adapters accept CUDA tensors on compute capability 9.x and 10.x, bfloat16
activations and expert weights, contiguous weights, and widths that are
multiples of 8; there `sonicmoe.moe.composed` (priority 110) is the semantic
default and `sonicmoe.moe` (95) the single-operator form. Both are
semantic-only: the explicit façade's residual tuple is the builtin's.
SonicMoE's expert-frequency offsets, gather/scatter/reverse indices and the
saved pre-activation are private custom-op residuals. The logical package
weights `w13 [E,2I,H]` (gate rows, then up rows) and `w2 [E,H,I]` are
SonicMoE's native `[E, out, in]` storage with its concatenated gate/up
layout, so the adapters take views and copy nothing. Quack is set up through
`mlops.quack_runtime` (patches, tuning cache, SonicMoE imports that keep
Quack's tuning policy).

Quack tunes each GEMM once per key (function, shapes, stride pattern,
dtypes) and, with `QUACK_CACHE_AUTOTUNING=1` (the runtime's default),
persists the choice under `QUACK_CACHE_DIR`; later processes read it. Two
things follow for anyone comparing runs across processes. Processes that
tune independently can choose different tile configs, which accumulate in a
different grouping and differ at an occasional bf16 rounding boundary. And
a process that ran the tuning sweep itself -- candidates compiled in Quack's
worker pool and benchmarked under CUDA-graph capture -- computes a slightly
different, deterministic result for the rest of its life than a process that
reads the cache, even for the same config. Both are valid roundings of the
same math; for bitwise agreement between processes, tune in a separate
process first (shadowspill's numerical gate runs a warm-up process before
its arms) and point every process at one cache directory. The file-by-file
structure is in
[Worked example: the SonicMoE implementations of `moe`](EXTENDING.md#worked-example-the-sonicmoe-implementations-of-moe)

FP8 expert weights are `QuackFP8Weight` containers, the expert-parallel
path's parameter type (`expert_parallel/quack/parameters/fp8.py`): each
weight's E4M3 rows and E4M3 transpose with one descale per row of each,
quantized once -- `QuackFP8Weight.quantize(w13, interleaved=True)` and
`QuackFP8Weight.quantize(w2)` from a checkpoint's weights, or `copy_` from a
bf16/fp32 master after an optimizer step (both orientations by the Triton
quantizers in `expert_parallel/quack/kernels/`). `w13`'s rows are stored
interleaved, gate and up rows adjacent, the order Quack's paired SwiGLU
epilogue reads; the logical view, `dequantize()` and the gradients stay in
`nn.Linear` order. With such weights `moe` runs the experts' GEMMs in E4M3 on
SM90 (`providers/sonicmoe/fp8.py`): the routed rows, the cotangents and the
rematerialized activation are quantized per row per call by the same
kernels, gathered or activated in the one pass; the up projection (with the
SwiGLU epilogue), the down projection and the two activation-gradient GEMMs
are the expert-parallel path's scale-aware epilogue kernels on the same
routing metadata as the bf16 forms, so the saved pre-activation `h` and the
residuals are unchanged. Weight gradients stay the bf16 varlen-K GEMMs, at
the containers' logical dtype. The operators take containers as the
expert-parallel operators do (a dispatch rule turns each into its
components). `compute_precision` names the GEMMs' precision apart from the
weights: `"fp8_current"` on bf16 weights quantizes them per call (straight
through for their gradients), `"bf16"` on containers dequantizes them per
call -- in every implementation, which is how the builtin and native
implementations take FP8 weights. SM100 is declined for FP8 math until the
kernels are validated there.

Block-scaled FP8 weights are `BlockFP8Weight` containers (`mlops/block_fp8.py`):
E4M3 stored once, in `nn.Linear` row order, with an FP32 scale per 128x128
block -- the layout of DeepSeek-style FP8 checkpoints. A block covers both
axes, so one copy serves the projections, which reduce over a weight's input
axis, and the input gradients, which reduce over its output axis; no
transposed copy is kept. They compute at `"fp8_block"` (their default) on
SM90 through Quack `GemmSm90` subclasses with their own producers and
mainloop (`expert_parallel/quack/experts/block_fp8_gemm.py`): 128x128x128
tiles over the same routing metadata, each K tile one scale block, its
wgmmas accumulated from zero and folded into the FP32 sum at the block's
scale. The rows are E4M3, quantized per row per call and gathered in the
same pass as for `"fp8_current"`, their scales applied in the epilogue.
Projections read the weight K-major, as stored; the up projection reads
`w13`'s gate and up rows as interleaved column pairs and writes `h` with gate
and up interleaved and the SwiGLU in its epilogue, as the two-copy form does.
Input gradients read it MN-major, which Hopper's E4M3 tensor cores do not
take, so a transposing warpgroup rewrites each tile in shared memory after
TMA loads it (FlashAttention-3's ldmatrix/prmt/stmatrix sequence) and the
cotangent rows are quantized in that transposition's K order -- the
interleaved `dh` read gate columns first; the down projection's input
gradient runs the SwiGLU backward in its epilogue. An up projection whose
gate and up halves are not whole 64-row groups runs on a grouped Triton
kernel (`kernels/block_fp8.py`) and the SwiGLU row kernel instead. Weight
gradients are the bf16 varlen-K GEMMs, as for the two-copy containers.

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
- [ ] Every registered custom operator, forward and backward, has a flop
      formula.
- [ ] Repeated non-unit backward preserves saved inputs and cotangents.
- [ ] Float64-capable paths pass registry-aware `gradcheck`; low-precision-only
      paths record an explicit skip and pass direct VJP parity.
- [ ] Forward, gradients, parameters, and optimizer state pass tensor metrics.
- [ ] Export/AOT retains the exact custom-op target.
- [ ] Runtime, invocation-local workspace, and allocator peak are measured.
- [ ] A many-step real-data canary records loss and throughput.

## Expert-parallel modules

`mlops.expert_parallel` contains optional QuackMoE, QuackMoELoRA, TEMoE and
TEMoELoRA modules with caller-owned groups and communication buffers.
See [expert-parallel configuration, installation and examples](EXPERT_PARALLEL.md).


## Hyper-connection implementations

`hyper_connection.py` defines the public `mhc_pre` and `mhc_combine` contracts.
Adapters live in `providers/native_torch/hyper_connection.py`,
`providers/builtin/hyper_connection.py`, and
`providers/tilekernels/hyper_connection.py`, and
`providers/deepgemm/hyper_connection.py`; private Triton code lives in
`kernels/hyper_connection.py` and `kernels/mhc_pre.py`. All adapters use the normal immutable registry,
metadata support predicates and per-operation overrides. Optional imports occur
only inside execution. Neither model nor ShadowSpill owns provider selection.

Preprocessing includes Sinkhorn but excludes the attention/MLP branch.
Postprocessing includes mixing, branch expansion and addition. The current
adapters expose the semantic/autograd surface; they do not advertise public
`mlops.explicit` entrypoints. Their private backward custom ops expose fake
outputs, canonical FLOP formulas, and first-order VJPs with tensor-only saved
state. Native preprocessing stays decomposable for recomputation. TileKernels
saves the small logits and reconstructs Sinkhorn iteration states on chip.

TileKernels is eligible only on SM90/SM100 and when its dependency is present.
Pre currently requires four streams and tokens divisible by eight; post requires
BF16 contiguous activations, four streams, hidden width divisible by 128, and
FP32 coefficient mode. SM90 has numerical/compiled forward/backward validation;
SM100 eligibility follows upstream support, without local hardware validation.
See [the operation contracts](OPS.md#hyper-connections) for exact provider IDs
and precision options. Unsupported cases select builtin/native implementations.

### DeepGEMM mHC preprocessing setup

The complete setup builds pinned DeepGEMM revision
`5081f8513af57e61696de4f4f04b0e707ffb503e` with its submodules against the
selected PyTorch/CUDA environment. `--no-deepgemm` skips this source build.
For an existing environment, run:

```bash
./scripts/setup_deepgemm.sh --python "$CONDA_PREFIX/bin/python"
```

The helper is idempotent; `--force` rebuilds. It requires CUDA >=12.9 and a
C++20 host compiler supporting `<format>` (for example GCC 13+). Set `CXX`
when the default compiler is older. DeepJIT also needs elfutils/libelf headers:
system development packages or `MLOPS_ELFUTILS_ROOT` are accepted. If neither is
available, the helper creates an isolated Conda build-dependency prefix.
`MLOPS_BUILD_DIR` controls source/build placement. TileKernels and TileLang are
installed by the ordinary `providers` extra, at their pinned/minimum versions.
DeepGEMM is built separately to avoid replacing the selected PyTorch dependency.

`deepgemm.mhc_pre` uses the existing semantic interface and first-order custom
op VJP, fake outputs and FLOP formulas. It is gated to SM90/SM100 and compatible
four-stream geometry; missing dependencies and unsupported geometry leave the
other providers available. See the precision, saved-state and backward
workspace description in [the operation contract](OPS.md#mhc_pre). SM100 is
supported by upstream kernels but has not been tested locally.
