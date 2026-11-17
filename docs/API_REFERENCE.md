# API index

Each public surface has one reference. Start with the [guides](README.md) for a
walkthrough, or use this table to find signatures, shapes and supported options.

| Namespace | Purpose | Reference |
| --- | --- | --- |
| `mlops` | Semantic tensor operations with ordinary autograd | [Operations](OPS.md) |
| `mlops.explicit` | Stateless forward/VJP calls and saved arguments | [Explicit operations](EXPLICIT_OPS.md) |
| `mlops.dispatch` | Selection, precision, support diagnostics, costs and gradcheck | [Dispatch](DISPATCH.md) |
| `mlops.optim` | `AdamW`, functional, destination-writing and in-place updates | [Optimizers](OPTIMIZERS.md) |
| `mlops.objectives` | Summed masked/weighted CE and clipped-policy formulas on target logprobs | [Objectives](OBJECTIVES.md) |
| `mlops.lora` | `LoRAConfig`, `LoRALinear`, `LoRAHead`, `apply_lora`, `parameter_report` | [LoRA](LORA.md) |
| `mlops.expert_parallel` | Quack/Transformer Engine MoE and LoRA layers | [Expert parallelism](EXPERT_PARALLEL.md) |
| `mlops.sequential_moe` | Bounded per-expert execution and LoRA | [Sequential experts](SEQUENTIAL_MOE.md) |
| `mlops.operator_metadata` | Axis, reduction, gradient, layout and preparation descriptions | [Compiler metadata](COMPILER_METADATA.md) |
| `mlops.quack_runtime` | Tuning choices and prepared-object ownership | [Tuning](sonicmoe-autotuning.md), [preparation](PREPARED_KERNELS.md) |

The [semantic quick index](OPS.md#quick-index) lists operations and output shapes.
The [explicit quick index](EXPLICIT_OPS.md#quick-index) lists manual VJPs and residuals.
[Attention](ATTENTION.md) explains provider geometry and precision choices for
linear/sparse attention, routing and residual mixing.

Private `torch.ops.mlops` targets serve capture and autograd. Models should use
the public functions above. Provider authors should read the
[registration and ownership rules](PROVIDERS.md).
Prepared-kernel factory import paths come from `KernelPreparation` declarations;
applications do not need to import private preparation targets.
