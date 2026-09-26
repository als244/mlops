# Optimizer API

This reference documents the ordinary PyTorch optimizer façade and the
stateless tensor transitions beneath it. All update tensors are contiguous
CUDA BF16, FP16, or FP32 arrays with one common element count. Optimizer
updates are deliberately non-differentiable.

## Contents

1. [`AdamW`](#adamw)
2. [`functional_adamw`](#functional_adamw)
3. [`adamw`](#adamw-1)
4. [`adamw_`](#adamw_)
5. [`functional_master_adamw`](#functional_master_adamw)
6. [`master_adamw`](#master_adamw)
7. [`master_adamw_`](#master_adamw_)
8. [Storage surfaces](#storage-surfaces)
9. [Arithmetic semantics](#arithmetic-semantics)

## `AdamW`

**Status:** Public

```python
AdamW(
    params,
    lr: float | Tensor = 1e-3,
    betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
    eps: float | Tensor = 1e-8,
    weight_decay: float | Tensor = 0.01,
    amsgrad: bool = False,
    *,
    maximize: bool = False,
    foreach: bool | None = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: bool | None = None,
    gradient_dtype: dtype | "parameter" = torch.bfloat16,
    reduction_dtype: dtype | "parameter" = torch.bfloat16,
    state_dtype: dtype | "parameter" = torch.bfloat16,
    parameter_rounding: Literal["nearest", "stochastic"] = "nearest",
    state_rounding: Literal["nearest", "stochastic"] = "nearest",
    replica_group=None,
    opt_state_strategy: Literal["replicated", "sharded"] = "sharded",
    gradient_reduction: Literal["sum", "mean"] = "mean",
    bucket_bytes: int = 64 << 20,
)
```

`params`, `lr`, `betas`, `eps`, `weight_decay`, `amsgrad`, `maximize`,
`foreach`, `capturable`, `differentiable`, and `fused` retain the normal
`torch.optim.AdamW` meanings. AMSGrad and differentiable updates are not
supported. `foreach` and `fused` are accepted as source/state-dict metadata;
choosing this class already selects the mlops implementation.

### Settings a step can change

`lr`, `betas`, `eps` and `weight_decay` are each accepted as a plain number or
as a scalar tensor, and are held in a host scalar either way -- once, where the
optimizer is built. A caller who means to change one between steps writes into
that tensor:

```python
optimizer = AdamW(parameters, lr=3e-4)
rate = optimizer.param_groups[0]["lr"]      # a host scalar
for step in schedule:
    with torch.no_grad():
        rate.fill_(step.rate)
    optimizer.step()
```

Two things follow, and both matter more under a captured step than an eager
one. The value is read where the kernel is launched, not where the update was
traced, so a captured step uses whatever was last written without being traced
again. And the scalars live on the host, so writing one copies nothing to the
device and synchronizes nothing -- the update passes the value as a launch
argument, which costs no kernel recompilation either.

They are held rather than converted per step for a reason worth stating: a
tensor made *inside* `step` would be made inside anything capturing that step
too, as an operation every parameter's update depends on. That reads as a
dependency between parameters that share nothing, and collapses a per-stage
update into a single task.

For the same reason, options are validated where they are set -- at
construction, and when a group is added -- rather than on every step: comparing
a setting held in a tensor is a branch on data a capture cannot resolve.

### dtype policies

The three dtype policies are independent and may be set globally or in an
individual parameter-group dictionary:

| Option | Meaning | Default |
|---|---|---|
| `gradient_dtype` | dtype used while packing a local gradient | BF16 |
| `reduction_dtype` | collective input/output and update-gradient dtype | BF16 |
| `state_dtype` | first- and second-moment dtype | BF16 |

`"parameter"` resolves to each parameter's storage dtype. The optimizer updates
each parameter at its own dtype and keeps no other copy of it: a master copy at
another precision belongs to whatever holds the training state, which hands the
optimizer the masters as its parameters -- with `gradient_dtype="parameter"`
for masters whose gradients arrive at their dtype. In sharded execution, each
rank retains only its update shard while the full model-visible parameter
remains available to forward computation.

`replica_group=None` selects local execution. Supplying a replica ProcessGroup
makes the optimizer own synchronization and selects
`opt_state_strategy="sharded"` by default; `"replicated"` is the other
supported state-ownership strategy. `gradient_reduction`
controls sum or mean semantics. `bucket_bytes` is a hard upper bound on each
full collective payload; an individual parameter may be split at arbitrary
element boundaries. Buckets with compatible dtype policies reuse task-lifetime
packing/reduction workspace, so live scratch is bounded by the largest bucket
per dtype class rather than the model size or number of buckets.

The same serialized algorithm supports NCCL and Gloo ProcessGroups. NCCL uses
a nonblocking current-stream completion bridge. Other backends use the
portable `Work.wait()` completion boundary; this is correct for Gloo's CUDA
reduce-scatter even though that Work type has no stream future in PyTorch 2.13.
Backend selection and completion policy are internal and expose no optimizer
option. The caller creates the replica group in the normal PyTorch way; NCCL
is the intended/default GPU training backend, while Gloo is a correct slower
fallback. Do not combine a group-backed optimizer with another wrapper that
also synchronizes the same gradients.

For local execution, each admitted parameter's standard optimizer state
contains scalar `step`, `exp_avg`, and `exp_avg_sq`. Distributed
state is stored in deterministic rank-local buckets because a parameter may
span buckets and sharded state is not a full per-parameter tensor. Frozen
parameters and parameters without local gradients are skipped by local
execution. `step()` returns the optional closure result and `synchronize()`
retires distributed work (a no-op locally).

The exact compiler identity is `builtin.adamw.triton`.

```python
optimizer = mlops.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

### Rounding

The update computes in FP32 and rounds what it stores to the dtype that holds
it. `parameter_rounding` chooses how the parameter is rounded, and
`state_rounding` how the first and
second moments are, each `"nearest"` (the default) or `"stochastic"`, globally
or per parameter group.

Stochastic rounding takes the representable neighbour on either side of the
value and picks the upper with probability equal to how far along the gap
between them the value lies, so the stored value is the computed one in
expectation. Rounded to nearest, an update smaller than half the gap to the
next value is lost every step; rounded stochastically it accumulates. This is
what lets a BF16 parameter, or BF16 moments, follow a trajectory that nearest
rounding would stall. It applies to BF16 and FP16 storage; FP32 storage holds
the computed value exactly. A value beyond the dtype's largest finite one is
rounded to nearest, and overflows as it would there.

The random bits are Philox bits counted from the step, a salt, and the
element's index, so a run is reproducible, and an update draws the same bits
eagerly and under anything capturing the step. Locally each parameter's salt
is its position among the optimizer's parameters. In distributed execution
each bucket has its own; in sharded execution each rank's shard also has its
own, and replicas updating a whole bucket draw the same bits, so they stay
replicas. The salt is a constant of each update, so under stochastic rounding
no two parameters' updates are the same to anything capturing the step;
rounded to nearest, every update is given the same salt.

Rounded stochastically, a stored value is rounded once. The
[PyTorch-compatible kernel](#arithmetic-semantics) otherwise rounds the
decayed parameter and the decayed second moment to nearest before the rest of
the update, as PyTorch's in-place primitives do; those roundings would bias
what stochastic rounding keeps.

## `functional_adamw`

```python
functional_adamw(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    step,
    *,
    gradient_scale=1.0,
    lr,
    betas,
    eps,
    weight_decay,
    maximize=False,
    parameter_rounding="nearest",
    state_rounding="nearest",
    rounding_salt=0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]
```

Returns new parameter, first moment, second moment, and step tensors. Inputs
are immutable. `gradient_scale` is applied in FP32 before the moment update.
`lr`, `betas`, `eps` and `weight_decay` each take a plain number or a host
scalar tensor, as described under
[Settings a step can change](#settings-a-step-can-change); the registered
operators take them as tensors, so a captured update reads them rather than
folding them in. `parameter_rounding` and `state_rounding` are as described
under [Rounding](#rounding); `rounding_salt` picks the stream of random bits
stochastic rounding draws from, and each tensor updated at a step needs its
own.

## `adamw`

```python
adamw(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    step,
    *,
    gradient_scale=1.0,
    lr,
    betas,
    eps,
    weight_decay,
    maximize=False,
    parameter_rounding="nearest",
    state_rounding="nearest",
    rounding_salt=0,
    out=(out_parameter, out_exp_avg, out_exp_avg_sq, out_step),
) -> tuple[Tensor, Tensor, Tensor, Tensor]
```

Writes a parameter-as-master update to storage-disjoint caller-owned `out`
tensors. Every destination matches its corresponding source's shape, stride,
dtype, and device. The function returns the same `out` tuple.

## `adamw_`

```python
adamw_(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    step,
    *,
    gradient_scale=1.0,
    lr,
    betas,
    eps,
    weight_decay,
    maximize=False,
    parameter_rounding="nearest",
    state_rounding="nearest",
    rounding_salt=0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]
```

Updates a parameter-as-master, moments, and step in place without allocating
tensors. A manual caller must prove that overwriting the old state is legal.

## `functional_master_adamw`

```python
functional_master_adamw(
    parameter,
    master_parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    step,
    *,
    gradient_scale=1.0,
    lr,
    betas,
    eps,
    weight_decay,
    maximize=False,
    parameter_rounding="nearest",
    state_rounding="nearest",
    rounding_salt=0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
```

Returns new model-parameter, master-parameter, first-moment, second-moment,
and step tensors. Inputs are immutable.

## `master_adamw`

```python
master_adamw(
    parameter,
    master_parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    step,
    *,
    gradient_scale=1.0,
    lr,
    betas,
    eps,
    weight_decay,
    maximize=False,
    parameter_rounding="nearest",
    state_rounding="nearest",
    rounding_salt=0,
    out=(
        out_parameter,
        out_master_parameter,
        out_exp_avg,
        out_exp_avg_sq,
        out_step,
    ),
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
```

Writes a distinct-master update to storage-disjoint caller-owned outputs and
returns the same `out` tuple.

## `master_adamw_`

```python
master_adamw_(
    parameter,
    master_parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    step,
    *,
    gradient_scale=1.0,
    lr,
    betas,
    eps,
    weight_decay,
    maximize=False,
    parameter_rounding="nearest",
    state_rounding="nearest",
    rounding_salt=0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
```

Updates the model parameter, distinct master parameter, moments, and step in
place without tensor allocation.

## Storage surfaces

| Surface | Preserves old state | Destination owner | Internal output allocation |
|---|---:|---|---:|
| `functional_adamw` / `functional_master_adamw` | yes | operation | yes |
| `adamw(..., out=...)` / `master_adamw(..., out=...)` | yes | caller | no |
| `adamw_` / `master_adamw_` | no | caller/optimizer | no |

All six tensor entrypoints share the same raw Triton mathematics. Their
registered functional, destination, and mutation targets have truthful
`mutates_args` and fake implementations. They do not register autograd because
an optimizer update is a terminal state transition, not a differentiable
model operation.

## Arithmetic semantics

The registered `builtin.adamw.triton` implementation is the
PyTorch-compatible kernel. It follows the ordered tensor primitives used by
default, non-fused `torch.optim.AdamW`: decoupled weight decay, `lerp` for the
first moment, a separately rounded multiply and `addcmul` for the second
moment, dtype-visible denominator construction, and the final `addcdiv`
update. It does not retain all intermediates in FP32 across those semantic
round points.

For the package default—BF16 parameters, gradients, moments, and no distinct
master—updates from normally zero-initialized optimizer state are bitwise
identical to default/foreach PyTorch AdamW in the permanent multi-step gate.
BF16, FP16, and FP32 first and second moments are also bitwise identical when
all update tensors share that dtype.

Parameter bitwise identity is intentionally not promised for every PyTorch
mode. Default non-capturable PyTorch computes bias-correction scalars from a
host step using Python floating-point arithmetic. mlops keeps the step and
update CUDA-resident so the operation remains asynchronous and suitable for
captured/lowered execution. Rare FP16 differences and FP32 last-bit
differences can therefore remain. `fused=True`, `capturable=True`, mixed dtype
policies, gradient scaling, a distinct master, and stochastic rounding are
separate arithmetic contracts rather than bitwise-equivalence claims.

The original kernel is retained in `mlops.kernels.adamw` as
`adamw_out_internal_fp32_raw` and
`adamw_master_out_internal_fp32_raw`. It keeps more intermediates in FP32 and
is a developer-only accuracy/performance ablation; registered operations and
`mlops.optim.AdamW` use the PyTorch-compatible kernel by default.
