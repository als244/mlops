# Extending mlops

This guide gives the required repository changes for each supported extension
path. Start with the smallest path that matches the desired behavior; a model
never needs to know which implementation was added.

## Contents

1. [Choose the extension](#choose-the-extension)
2. [What mlops considers an operation](#what-mlops-considers-an-operation)
3. [Rules shared by every extension](#rules-shared-by-every-extension)
4. [Add an implementation for an existing operation](#add-an-implementation-for-an-existing-operation)
5. [Add a new semantic operation](#add-a-new-semantic-operation)
6. [Other supported extensions](#other-supported-extensions)
7. [Validation matrix](#validation-matrix)
8. [Completion checklist](#completion-checklist)

## Choose the extension

| Goal | Primary files | Public API change? |
|---|---|---|
| New provider or implementation for an existing operation | `src/mlops/providers/<source>/<operation>.py`, provider catalog | No |
| New implementation variant from an existing provider | existing provider module or a focused sibling module | No |
| New mathematical operation | semantic façade, at least one provider, exports, docs, usually explicit module | Yes |
| Add explicit forward/VJP access to an existing operation | `src/mlops/explicit/<operation>.py`, provider entrypoints, docs | Yes, explicit API only |
| Add or improve cost hints | operation estimator and/or `Implementation.estimate` | No execution change |
| Add a raw Torch/Triton kernel | `src/mlops/kernels/<operation>.py` plus an implementation adapter | No by itself |
| Add an optional dependency | provider adapter, support gate, project optional dependency | No |
| Add dispatch/development infrastructure | `src/mlops/dispatch/`, exports, API docs | Contributor API only |
| Add a standard optimizer | `src/mlops/optim/`, state-update provider, raw kernel, optimizer API docs | Yes, optimizer API |

Provider is source metadata. Selection and artifacts use an exact
`implementation_id`, so one provider may implement one operation, several
operations, or several variants of an operation.

## What mlops considers an operation

An operation is a stable, stateless tensor computation or effect boundary with
a provider-independent mathematical contract. It is **not** defined by kernel
count, source language, or graph size. One operation may lower to one ATen
node, several Torch operations, one fused Triton kernel, a forward/backward
library pair, or a structured multi-kernel algorithm.

Operations can therefore exist at several compositional levels:

| Level | Examples | Why it is one operation |
|---|---|---|
| Atomic tensor primitive | `rms_norm`, `gelu`, `rope`, `cross_entropy` | One reusable mathematical transform with a compact VJP |
| Fused local composition | `gated_rms_norm`, `head_loss` | Fusion materially changes temporary memory or launch behavior while preserving a stable mathematical result |
| Sequence algorithm | `causal_conv_silu`, `linear_attention`, `flash_attention` | The boundary owns sequence-reset semantics and may map to a library-managed multi-kernel forward/backward algorithm |
| Structured/composite operation | `moe` | Routing, routed expert execution, auxiliary statistics, residual/shared-expert composition, and their gradients form one provider-independent contract |
| Non-differentiable state update | `adamw` | Parameter/moment/step transition has stable functional, destination, and mutation semantics plus an exact implementation identity |
| Semantic policy/composition | `linear_mixer` | A stable callable contract chooses packed or per-sequence organization without exposing that policy to model code |

The FLA-backed operations demonstrate that a library call may be an opaque
interior region while still composing normally with surrounding operations.
The `moe` operation demonstrates the opposite extreme: it is deliberately
larger than one grouped GEMM because callers need one stable routed-MoE
contract, while builtin, ScatterMoE, and a future SonicMoE implementation may
use different sorting, scatter/gather, residual, and VJP mechanics internally.
That semantic scope does not require one opaque dispatcher target. The
default composed builtin keeps the single public `moe` contract while
exposing a preparation region (routing/dispatch/`w13`) and a finish region
(SwiGLU/`w2`/combine), allowing AOT to replay only the residual-producing
prefix. Use this pattern when a provider is physically multi-kernel and the
split improves lifetime or rematerialization; keep a genuinely fused provider
opaque.

An operation is not:

- an `nn.Module` that owns parameters;
- a transformer block or complete model;
- automatically a `torch.library.custom_op`;
- automatically one runtime task or stage; or
- every raw kernel that happens to exist under `src/mlops/kernels/`.

Use a new operation when callers need a new mathematical contract. Use a new
implementation when the public arguments, outputs, and semantics remain the
same but the execution strategy changes. Keep ordinary compositions outside
the registry when they do not need an independently selectable implementation,
explicit VJP, cost boundary, or opaque compiler target.

## Rules shared by every extension

1. Keep models on the semantic API. They must not import providers, kernels,
   dispatcher targets, or explicit VJPs.
2. Keep the registry stateless. It may retain callables and immutable scalar
   metadata, never invocation tensors.
3. Make every tensor dependency an argument, output, ordinary autograd save,
   module-owned buffer, or caller-owned reusable object.
4. Treat workspace as scratch whose lifetime ends with one forward or backward
   call. Inputs, outputs, autograd residuals, tables, and prepacked weights are
   not workspace.
5. Do not silently fall back inside an implementation. Reject unsupported
   inputs through `SupportResult.no(reason)` and let the resolver select.
6. Preserve caller inputs, saved residuals, and cotangents. Registered training
   boundaries are functional unless their schema truthfully says otherwise.
7. Use native PyTorch for the correctness graph when possible. Use a registered
   custom op only when an opaque kernel/library boundary is needed.
8. Keep the semantic ABI mathematical. Provider-specific launch metadata and
   private residuals belong in the provider adapter, not a universal `extra`.
9. Keep entrypoints asynchronous. Do not read a device tensor back to Python
   to derive counts, offsets, or launch metadata. Prepare reusable values from
   host round/configuration data and pass caller-owned tensors explicitly.
   Gate development with `torch.cuda.set_sync_debug_mode("error")`, warmed
   enqueue-time measurements, and a profiler trace because PyTorch's detector
   is intentionally incomplete.

See [PROVIDERS.md](PROVIDERS.md) for the complete ownership, custom-op, and
support contracts.

## Add an implementation for an existing operation

Use this path for a new package such as Liger, a new hardware-specific kernel,
or another exact variant from an existing source.

### 1. Read and preserve the operation contract

Find the operation in [OPS.md](OPS.md). Match its argument order, shapes,
dtypes, semantic outputs, gradients, mutation rules, and exceptions. Private
custom-op outputs may differ, but `Implementation.apply` must return exactly
the documented semantic result.

### 2. Choose the provider location and identity

Create:

```text
src/mlops/providers/<source>/<operation>.py
```

Use an exact stable ID such as:

```text
<source>.<operation>.<variant>
```

Omit `<variant>` only when the provider has one unambiguous implementation.
The ID, not provider name, is used for overrides and serialized manifests.

### 3. Adapt raw stateless forward and backward functions

Call a third-party low-level API directly, or place package-owned mechanics in
`src/mlops/kernels/<operation>.py`. The adapter should expose ordinary
argument-driven functions:

```python
def forward(x, weight, eps):
    return output, residual


def backward(grad_output, x, weight, residual):
    return grad_x, grad_weight
```

Do not instantiate a parameter-owning vendor module when caller-owned tensors
can be passed directly. Do not store residuals on the provider object.

### 4. Write conservative support detection

Support checks inspect metadata only and explain every rejection:

```python
def _supports(x, weight, eps=1e-5, *, surface, **_kwargs):
    if dependency_is_missing:
        return SupportResult.no("package-name is unavailable")
    if not x.is_cuda:
        return SupportResult.no("requires CUDA")
    if x.dtype not in supported_dtypes:
        return SupportResult.no("unsupported activation dtype")
    return SupportResult.yes()
```

Check dependency/version, device architecture, dtype, layout, dimensions, and
provider restrictions. `surface` is either `"semantic"` or `"explicit"`; use
it when only one surface is supported.

### 5. Choose a semantic boundary

For a captureable native-PyTorch graph, `apply` can call the graph directly.

For opaque Triton, CUDA, or third-party calls, register a functional boundary:

1. `torch.library.custom_op("mlops::<stable_target>_fwd", mutates_args=())`;
2. a fake implementation with identical output metadata;
3. `setup_context`, saving only custom-op inputs or outputs;
4. `register_autograd` on the forward target; and
5. a separate registered backward target when the VJP is also opaque.

The semantic `apply` calls the registered forward target and hides any private
residual outputs. Use
`src/mlops/providers/builtin/rms_norm.py` as the small opaque example and
`src/mlops/providers/scattermoe/moe.py` as the composite example.

### 6. Expose explicit entrypoints when supported

Set `Implementation.forward` and `Implementation.backward` to the same raw
functions used by the custom-op adapter. Set both or neither. An apply-only
implementation is valid, but exact explicit selection will reject it.

Do not add provider arguments to the public explicit operation. Provider-only
residuals can remain private to the registered custom-op path. If the existing
explicit ABI cannot represent the new provider without generic metadata,
prefer an apply-only implementation and document why.

### 7. Add optional cost hints

`Implementation.estimate` may report implementation FLOPs, bytes accessed, and
invocation-local workspace. Return `None` for unknown, never a guessed zero.
Do not change the canonical logical estimator merely because one provider does
extra work.

### 8. Register the implementation

Create one immutable record:

```python
IMPLEMENTATION = register_implementation(
    Implementation(
        operation="rms_norm",
        implementation_id="example.rms_norm.variant",
        provider="example",
        priority=50,
        deterministic=True,
        supports=_supports,
        apply=apply,
        forward=forward,
        backward=backward,
        estimate=_estimate,
    )
)
```

Add the module string to `_CATALOG` in
`src/mlops/providers/__init__.py`. Optional dependency import failures must
make only this implementation unavailable.

### 9. Document and test it

Update the exact implementation table in [OPS.md](OPS.md) and document any
constraints or numerical semantics. Add tests for:

- support and forced-rejection explanations;
- semantic forward and gradients;
- explicit forward and arbitrary-cotangent VJP, when exposed;
- fake metadata and `torch.library.opcheck`, when opaque;
- repeated retained-graph backward and input/cotangent immutability;
- exact Export/AOT target retention;
- registry-aware gradcheck or an explicit low-precision skip reason; and
- representative fidelity, throughput, and allocator/workspace measurements.

### Worked example: add SonicMoE to `moe`

SonicMoE is an implementation of the existing `moe` operation, not a new
semantic operation. The reviewed interface can consume either fused routing or
caller-supplied sorted routing, while the current `moe` ABI already defines
hidden/residual inputs, logical router and expert weights, routing policy,
top-k, packed lengths, shared-expert inputs, auxiliary statistics, and all
required gradients.

The first mlops adapter must support **only SM90 and SM100**, represented by
CUDA compute-capability tuples `(9, 0)` and `(10, 0)`. This is a deliberately
conservative mlops validation boundary. It does not imply that the upstream
project supports no other architecture; each additional architecture must be
enabled here only after its own correctness, capture, and performance results
are recorded.

The integration would proceed as follows.

1. **Keep the public ABI unchanged.** Do not add Sonic-specific offsets,
   gather indices, transposed weights, or workspace arguments to
   `mlops.moe(...)` or `mlops.explicit.moe`.
2. **Add the optional dependency.** Add a pinned `sonicmoe` optional dependency
   in `pyproject.toml`. An unavailable or incompatible build must reject only
   `sonicmoe.moe`.
3. **Create the provider package.** Add
   `src/mlops/providers/sonicmoe/__init__.py` and
   `src/mlops/providers/sonicmoe/moe.py`.
4. **Preserve canonical routing first.** Reuse the package's existing routing
   policy and auxiliary-statistic semantics, then adapt the resulting token-to-
   expert assignments to SonicMoE's caller-supplied sorted-routing entrypoint.
   Use SonicMoE's fused-router path only if it can be proven to reproduce the
   selected routing mode, top-k weights, bias/group rules, forced decisions,
   and auxiliary outputs exactly.
5. **Build private routing metadata inside the adapter.** Expert-frequency
   offsets, sorted assignment indices, gather/scatter indices, and reverse-
   scatter metadata are provider-private forward outputs or invocation-local
   workspace. The registry must not cache them.
6. **Adapt logical expert weights without changing model code.** The canonical
   `w13[E,H,2I]` and `w2[E,I,H]` tensors correspond to SonicMoE-style
   `[2I,H,E]` and `[H,I,E]` views. Accept the provider only when those views and
   strides are supported. If packing is required, define a separate explicit
   caller-owned preparation contract before enabling that geometry; do not
   hide a persistent packed copy in the provider.
7. **Register an opaque forward target.** A private
   `mlops::moe_sonicmoe_fwd` custom op calls the low-level SonicMoE route/expert
   engine and returns the normal semantic results plus only the private tensors
   needed by its VJP. Its fake function reproduces every output's metadata.
8. **Register the VJP.** `setup_context` saves the required custom-op inputs and
   private outputs. A functional `mlops::moe_sonicmoe_bwd` target returns hidden,
   router, expert-weight, route-weight, and shared-expert gradients in the
   canonical order without mutating saved values or cotangents.
9. **Expose the existing explicit ABI if possible.** Implement the current
   explicit MoE forward/backward signatures only if SonicMoE can consume the
   documented residual tuple efficiently. Otherwise register it as
   semantic/apply-only; do not weaken the explicit ABI with a generic extra.
10. **Register exact identity and support.** Add an `Implementation` with
    `operation="moe"`, `implementation_id="sonicmoe.moe"`, and
    `provider="sonicmoe"`. Its architecture check must accept exactly
    `{(9, 0), (10, 0)}` and return a specific rejection such as
    `"sonicmoe.moe requires SM90 or SM100"` for every other capability. Also
    gate on dependency version, dtype/layout, expert dimensions, top-k,
    routing mode, and supported weight strides. Add the module to `_CATALOG`.
11. **Report physical costs.** Keep the existing logical MoE estimate. The
    SonicMoE estimator reports known gather/scatter traffic and only
    invocation-local temporary storage as workspace; unknown fields remain
    `None`.
12. **Validate before changing priority.** Compare builtin and SonicMoE outputs
    and all differentiable inputs across supported routing modes, packed
    lengths, shared/residual experts, top-k values, and forced routing. Then run
    opcheck, repeated backward, capture, many-step fidelity, throughput, and
    allocator-peak tests. Keep builtin selection as the default until the
    measured result justifies promotion.

This pattern is the key reason the public MoE contract does not standardize one
provider's saved routing tuple: implementations remain free to use the metadata
and fused layout their execution engine needs.

### Worked example: one QuACK provider for RMSNorm and cross entropy

[QuACK](https://github.com/Dao-AILab/quack) demonstrates that a provider is not
an operation and does not own a single global dispatch choice. One optional
package can register any subset of implementations independently. In this
example it contributes `quack.rms_norm` and `quack.cross_entropy` to two
existing exact operations.

Cross entropy became a standalone operation when mlops was separated from its
first model consumer. Earlier code had only private cross-entropy mechanics
inside the chunked head. The public `cross_entropy` contract now owns reusable
logits-to-loss math, while `head_loss` remains a larger bounded-logits
projection/loss operation. A QuACK cross-entropy kernel cannot substitute for
`head_loss` because it does not own projection or chunking.

As with SonicMoE, the initial QuACK adapters must support **only SM90 and
SM100**, represented by `(9, 0)` and `(10, 0)`. Upstream may expose other
architectures; mlops intentionally rejects them until they are independently
validated for each registered operation.

The integration would proceed as follows.

1. **Define the two registration outcomes.** Both are implementations of
   existing operations:

   ```text
   provider  operation      implementation_id
   quack     rms_norm       quack.rms_norm
   quack     cross_entropy  quack.cross_entropy
   ```

   The records share provider metadata and dependency checks but resolve,
   override, trace, estimate, and fail independently.
2. **Add one optional dependency and focused provider package.** Pin the
   reviewed `quack-kernels` release in a `quack` optional-dependency group and
   create:

   ```text
   src/mlops/providers/quack/
   ├── __init__.py
   ├── _support.py
   ├── rms_norm.py
   └── cross_entropy.py
   ```

   `_support.py` may contain immutable version and hardware checks shared by
   these adapters. It must not contain a provider-global selected
   implementation or retain invocation tensors.
3. **Implement the shared architecture gate.** On a concrete CUDA warmup,
   inspect `torch.cuda.get_device_capability(tensor.device)` and accept exactly
   `{(9, 0), (10, 0)}`. CPU, SM80, SM120, and unknown capabilities reject with
   an operation-specific reason. Keep dtype, stride, width, and package-version
   checks in each operation module because those constraints may differ:

   ```python
   _SUPPORTED_QUACK_CAPABILITIES = frozenset({(9, 0), (10, 0)})

   def _require_quack_architecture(operation, tensor):
       capability = torch.cuda.get_device_capability(tensor.device)
       if capability not in _SUPPORTED_QUACK_CAPABILITIES:
           return SupportResult.no(
               f"quack.{operation} requires SM90 or SM100; got SM"
               f"{capability[0]}{capability[1]}"
           )
       return SupportResult.yes()
   ```

   Exact overrides still run this check and fail; they must never bypass the
   gate or fall back.
4. **Adapt QuACK RMSNorm's low-level pair.** Call its `rmsnorm_fwd` with the
   package's documented Llama semantics: affine weight, no bias/residual,
   caller `eps`, stored fp32 `rstd`, and weight offset zero. Call
   `rmsnorm_bwd` with `x`, `weight`, `grad_output`, and `rstd`. Do not use
   QuACK's `nn.Module`, monkey patching, or `RMSNormFunction` as the mlops
   boundary.
5. **Own an mlops RMSNorm custom-op pair.** Register functional
   `mlops::rms_norm_quack_fwd` and `mlops::rms_norm_quack_bwd` targets with fake
   functions and registered autograd. Their adapter calls the same raw
   functions exposed through `explicit.rms_norm`; the public RMSNorm signatures
   and residual `(output, rstd)` remain unchanged.
6. **Preserve the canonical cross-entropy contract.** Match the existing
   semantic contract:

   ```python
   cross_entropy(
       logits,                 # [R, V], floating point
       targets,                # [R], integer class IDs
       *,
       weight=None,            # optional [V], non-differentiable class weights
       ignore_index=-100,
       reduction="mean",       # "none", "sum", or "mean"
   ) -> loss                  # [R] for "none", scalar otherwise
   ```

   Mean reduction uses valid weighted targets and `logits` is the only
   differentiable tensor. Use `native_torch.cross_entropy` and
   `builtin.cross_entropy.triton` as independent correctness/fidelity oracles.
   The contract remains distinct from bounded `head_loss`.
7. **Define a flat explicit cross-entropy VJP.** The provider-independent
   explicit surface can expose the backward residual QuACK naturally computes:

   ```python
   losses, logsumexp = explicit.cross_entropy.forward(
       logits, targets, weight, ignore_index
   )
   grad_logits = explicit.cross_entropy.backward(
       grad_losses, logits, targets, logsumexp, weight, ignore_index
   )
   ```

   Here `losses`, `logsumexp`, and `grad_losses` have shape `[R]`; reduction is
   ordinary semantic Torch math around this unreduced kernel boundary. This
   keeps the explicit VJP flat, makes every saved tensor visible to its caller,
   and lets arbitrary cotangents test the kernel. It does not add a generic
   provider `extra`.
8. **Adapt QuACK cross entropy without mutation.** Use its low-level
   `cross_entropy_fwd(..., return_lse=True, inplace_backward=False)` and
   `cross_entropy_bwd(..., inplace_backward=False)` behavior. Do not expose
   QuACK's optional partial-LSE mode in the canonical operation until another
   public contract requires distributed vocabulary partitioning. Do not use
   its `CrossEntropyFunction` as the mlops compiler boundary.
9. **Own a second mlops custom-op pair.** Register
   `mlops::cross_entropy_quack_fwd` returning unreduced fp32 losses and fp32
   log-sum-exp, plus `mlops::cross_entropy_quack_bwd` returning logits-dtype
   gradients. `setup_context` saves ordinary references to logits, targets,
   log-sum-exp, and optional class weights. The adapter remains stateless;
   standard autograd owns these saved tensors until the matching backward.
10. **Register both records independently.** `quack.rms_norm` and
    `quack.cross_entropy` each receive their own priority, deterministic flag,
    support function, semantic `apply`, explicit pair, and optional estimator.
    Add both modules to `_CATALOG`. A shape unsupported by QuACK cross entropy
    must not disable QuACK RMSNorm, and selecting one must not select the other.
11. **Report operation-specific costs.** RMSNorm reports its norm traffic and
    backward reduction scratch. Cross entropy reports row/vocabulary work,
    logits/target/loss traffic, and only invocation-local CuTe scratch as
    workspace. Log-sum-exp is an autograd residual/output, not workspace.
12. **Validate the provider as two integrations.** For both operations, cover
    forced support rejection on CPU, SM80, and SM120; acceptance on mocked
    SM90/SM100 metadata; forward/VJP parity; fake metadata; opcheck; repeated
    backward; immutability; Export/AOT target retention; and cost hints. Run
    float64 gradcheck only on a capable oracle implementation; QuACK's
    low-precision CUDA paths use direct arbitrary-cotangent VJP comparisons.
    Then benchmark each operation independently and together in a model before
    changing either priority.

This example establishes two useful rules: a provider directory may register
multiple operations, and shared availability checks do not imply shared
selection. It also records the earlier API lesson: private kernel mechanics do
not become a reusable public operation until a canonical contract, provider
records, explicit VJP, tests, and documentation exist.

## Add a new semantic operation

This path changes the public API and therefore starts with the mathematical
contract, not with an available kernel. Apply the definition in
[What mlops considers an operation](#what-mlops-considers-an-operation): a new
vendor, fusion strategy, or private residual layout is normally a new
implementation, not a new operation.

### 1. Define the canonical ABI

Before coding, add or draft the [OPS.md](OPS.md) entry:

- semantic name and complete signature;
- tensor shapes, dtypes, devices, and scalar metadata;
- ordered public outputs;
- differentiable and non-differentiable arguments;
- mutation and alias behavior;
- required first-order VJP semantics;
- packed-sequence/reset behavior, if applicable; and
- errors and unsupported cases.

Do not expose vendor launch choices or a generic provider `extra` in this ABI.

### 2. Add the semantic façade

Create `src/mlops/<operation>.py` with validation that is common to every
implementation and one resolver call:

```python
def operation(x, weight, option=False):
    implementation = resolve_implementation(
        "operation", x, weight, option, surface="semantic"
    )
    return implementation.apply(x, weight, option)
```

Export the symbol from `src/mlops/__init__.py`. Keep kernel branches and
provider imports out of this file.

### 3. Add a native-PyTorch correctness implementation

Create `src/mlops/providers/native_torch/<operation>.py` whenever the math can
be expressed in PyTorch. This implementation establishes readable semantics,
CPU/float64 coverage where meaningful, and a gradcheck oracle. Register it
with lower priority than accelerated defaults.

If PyTorch cannot express the operation, document that limitation and provide
an independent numerical oracle in tests.

### 4. Decide whether the operation has an explicit API

Add `src/mlops/explicit/<operation>.py` when the operation is a useful atomic
forward/VJP boundary. Export exactly `forward` and `backward`, name every saved
argument explicitly, call resolved raw functions under `torch.no_grad()`, and
return gradient values without touching `.grad`.

Export the module from `src/mlops/explicit/__init__.py`. Composite helpers,
discrete selections, and layout/control policies may remain semantic-only;
record the reason in [OPS.md](OPS.md).

### 5. Add accelerated implementations

Follow [Add an implementation for an existing operation](#add-an-implementation-for-an-existing-operation)
for each builtin or third-party path. Register a custom op only for opaque
boundaries; visible ATen graphs do not need one.

### 6. Add logical cost hints

Register at most one canonical operation estimator with
`register_operation_estimator`. It describes mathematical FLOPs and minimum
bytes, independent of provider. Concrete implementations may overlay physical
cost and workspace through their `estimate` functions.

### 7. Wire discovery and documentation

Add every implementation module to `src/mlops/providers/__init__.py`. Then
update:

- semantic and explicit quick indexes in [OPS.md](OPS.md);
- the appropriate operation-family section in [OPS.md](OPS.md);
- the category matrix in [API_REFERENCE.md](API_REFERENCE.md); and
- `docs/EXPLICIT_OPS.md` if the new explicit surface introduces a new pattern.

### 8. Complete public-contract tests

In addition to provider tests, verify that:

- `mlops.__all__` and, when applicable, `mlops.explicit.__all__` include it;
- its signature and every parameter appear in [OPS.md](OPS.md);
- semantic modules contain no custom-op registration or direct kernels;
- every implementation is deterministically discoverable;
- unsupported exact overrides fail rather than fall back; and
- the operation composes through normal autograd, Export, and AOTAutograd.

## Other supported extensions

### Add a variant from an existing provider

Register a distinct `implementation_id`; never overload an existing ID based
on hidden environment state. Put a substantial variant in a sibling file when
that keeps support logic and custom-op ownership clearer.

### Add explicit access to an existing operation

Define one provider-independent residual ABI, add
`src/mlops/explicit/<operation>.py`, and expose raw forward/backward functions
from every implementation that can satisfy it. Apply-only implementations may
remain registered with clear explicit-surface rejection reasons.

### Add or refine cost hints

Canonical estimators describe provider-independent logical work. Exact
implementation estimators describe physical work and workspace. They must
inspect metadata without running kernels or retaining tensors. Profiling
remains authoritative.

### Add a raw kernel

A file under `src/mlops/kernels/` is private mechanics, not a selectable
implementation. Add or update a provider adapter before the kernel is reachable
through public APIs. Raw kernels never resolve implementations or fall back.

### Add an optional dependency

Add it to the appropriate optional-dependency group in `pyproject.toml`, import
it defensively in its provider module, expose its installed version in support
diagnostics, and ensure an absent dependency does not break catalog bootstrap.

### Add dispatch or diagnostic functionality

This is intentionally rare. Keep dispatch APIs tensor-stateless and
context-local, export contributor-facing symbols from `mlops.dispatch`, add
them to [API_REFERENCE.md](API_REFERENCE.md), and prove nested contexts restore
correctly. Diagnostic controls must not become hidden compiled-artifact state.

## Validation matrix

| Change | Required minimum validation |
|---|---|
| Existing op, native graph implementation | semantic parity, gradients, support/override behavior, gradcheck, Export/AOT |
| Existing op, opaque implementation | all native checks plus fake metadata over model-authentic contiguous/non-contiguous layouts, opcheck, runtime output-layout comparison, target retention, repeated backward, immutability |
| Existing op, explicit entrypoints | explicit forward and arbitrary-cotangent VJP parity, no autograd recording, no `.grad` writes |
| New semantic operation | public export/docs coverage, native oracle, every provider, semantic/explicit/AOT composition |
| Cost-only change | exact arithmetic tests, undefined-versus-zero behavior, no tensor retention or kernel execution |
| Raw-kernel-only change | owning provider tests; raw kernel must remain unreachable without an adapter |
| Optional dependency | installed and missing-dependency cases |

Run the complete standalone gates after every extension:

```bash
python -m pytest -q tests
ruff check src/mlops tests
```

## Completion checklist

- [ ] Public mathematical contract is unchanged or deliberately documented.
- [ ] Semantic call sites contain no provider-specific logic.
- [ ] Exact IDs, priorities, determinism, and support reasons are explicit.
- [ ] Implementations of an operation that takes a determinism request accept
      the `deterministic` keyword, and honour it or raise.
- [ ] Registry, provider, trace, and estimator state retain no tensors.
- [ ] Raw functions are stateless and do not silently fall back.
- [ ] Custom-op schema, fake outputs, and registered autograd agree.
- [ ] Fake/runtime output shape, dtype, device, stride, storage offset, and aliases agree for authentic caller layouts.
- [ ] Explicit entrypoints share raw math and expose all caller-owned residuals.
- [ ] Inputs, residuals, and cotangents survive repeated backward unchanged.
- [ ] Logical and physical cost hints do not confuse state with workspace.
- [ ] API references, quick indexes, examples, and contributor docs are updated.
- [ ] Standalone tests, documentation checks, and lint pass.
