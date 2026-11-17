# Package architecture

MLOps owns operations and local optimizer updates. Models, training loops,
global state sharding and execution schedules belong to callers.

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
| `src/mlops/kernels/` | raw Torch/Triton/TileLang mechanics; no selection or fallback |
| `src/mlops/objectives/` | model-independent summed token formulas and row derivatives; fused head kernels live in `kernels/objectives/` |
| `src/mlops/optim/` | standard optimizers plus functional, `out=`, and in-place update entrypoints |
| `src/mlops/preparation/` | graph-visible sequence metadata plus explicit external-kernel build/load factories |
| `src/mlops/operator_metadata/` | standalone axis, reduction, gradient, layout and preparation declarations; consuming compilers own their adapters |
| `src/mlops/triton_runtime/` | prepared external Triton calls, validated binaries and owned CUDA modules |
| `src/mlops/lora/` | LoRA configuration, model-independent conversion and low-rank modules |
| `src/mlops/expert_parallel/`, `src/mlops/sequential_moe/` | optional MoE layer packages with their own documentation |
| `src/mlops/kernels/kda/`, `kernels/sparse_mla/` | attributed attention kernels, imported lazily by standard registrations |
| `src/mlops/quack_runtime/` | process-wide Quack setup shared by the expert-parallel layers and the SonicMoE provider: source-checked patches, the tuning-cache default, kernel-source fingerprint registration, SonicMoE imports that keep Quack's tuning policy, and task-local choices/caches with owned prepared objects |
| `src/mlops/bootstrap.py` | registers custom-op targets before artifact loading: the implementation catalog's and the LoRA, router and sequential-MoE operators; expert-parallel operators register when the modules defining them are imported, and accessing a layer class imports its own |

`provider` is source metadata (`builtin`, `native_torch`, `fla`, `cula`,
`cudnn`, `deepgemm`, `tilekernels`, or `sonicmoe`). Selection always uses an
exact implementation ID such as `builtin.rms_norm.triton`; there is no
provider-wide switch.

## Tensor and kernel ownership

Inputs and reusable metadata belong to callers. Outputs belong to the caller
after return; autograd retains only the residuals needed by its matching VJP.
Workspace exists only within an invocation. Registries and dispatch traces
must not retain tensors. See the [provider ownership rules](PROVIDERS.md#tensor-ownership).

Opaque differentiable calls declare a custom-op schema, matching fake result
and autograd/VJP. A composition can expose several opaque regions so a compiler
can regenerate only the necessary prefix. Explicit calls use the same raw math
under `torch.no_grad()` and never write `.grad`.

Optimizer updates use local tensors; the caller owns communication and shard
placement. Functional, destination-writing and in-place forms share the
[optimizer implementation](OPTIMIZERS.md). Destination-writing operations
declare all mutation and require the caller to establish legal ownership.

## Compiler integration boundary

[Operator metadata](COMPILER_METADATA.md) lives in MLOps because provider authors
own their semantics. They contain no imports or types from an execution engine.
The consuming project translates these facts into its own build controls and
owns integration tests. No compiler adapter is installed in this package.

[Prepared-kernel factories](PREPARED_KERNELS.md) own external source checks,
autotuning choices, host launchers and device-module lifetimes. Their manifests
are opaque to callers. These compiled resources are distinct from invocation
tensors: preparation does not retain model activations, choose checkpoint cuts
or implement execution-region chunking. Destination-writing backward overloads
explicitly declare mutation and are used only when the caller proves it legal.

## Further reading

- [Dispatch](DISPATCH.md): exact selection, precision, cost hints and diagnostics.
- [Provider requirements](PROVIDERS.md): schemas, fake layouts, residuals and async behavior.
- [Extension guide](EXTENDING.md): implementation and operation walkthroughs.
- [Expert parallelism](EXPERT_PARALLEL.md): optional groups, communication and GPU validation.
- [Attention](ATTENTION.md): geometry, precision and kernel attribution.
