# MLOps documentation

Start with [installation](INSTALLATION.md) and the [small tensor example](../README.md).
Use the guides for a task and the references for exact signatures and constraints.

## Use the library

| Task | Read |
| --- | --- |
| Find a function, dtype or tensor shape | [Semantic operations](OPS.md) |
| Choose a provider, precision or determinism setting | [Dispatch](DISPATCH.md) |
| Call a VJP and manage saved state | [Explicit operations](EXPLICIT_OPS.md) |
| Configure low-rank adapters | [LoRA](LORA.md) |
| Use fused CE, SFT, policy or sampled-distillation losses | [Objectives](OBJECTIVES.md) |
| Configure AdamW, masters or gradient precision | [Optimizers](OPTIMIZERS.md) |
| Use attention, routing or residual mixing | [Attention guide](ATTENTION.md) |
| Distribute experts or bound expert workspace | [Expert parallelism](EXPERT_PARALLEL.md), [sequential experts](SEQUENTIAL_MOE.md) |
| Configure GEMM tuning and caches | [SonicMoE autotuning](sonicmoe-autotuning.md) |

The [API index](API_REFERENCE.md) maps each public namespace to its reference.

## Extend or integrate

Read [architecture](ARCHITECTURE.md) for ownership and call flow, then choose:

- [Extension guide](EXTENDING.md): add an operation or implementation, with worked examples.
- [Provider requirements](PROVIDERS.md): registration, tensor ownership, fake/autograd behavior and asynchronous calls.
- [Raw kernels](KERNELS.md): kernel boundaries, attribution and dependencies.
- [Compiler metadata](COMPILER_METADATA.md): independent axes, reductions, layout and gradient destinations.
- [Prepared kernels](PREPARED_KERNELS.md): export, restore and own external compiled resources.

## Maintain the docs

Keep model-facing signatures in `OPS.md`, manual VJPs in `EXPLICIT_OPS.md`, and
selection controls in `DISPATCH.md`. Link those references from guides instead
of repeating their rules. Architecture explains ownership; setup commands
belong in `INSTALLATION.md`. Performance experiments and review notes belong
outside the public manuals.

Run `python -m pytest -q tests/test_documentation.py` after editing. It checks
the complete public Markdown set, local links, anchors, tables and exported
operation signatures. Hardware examples still need their corresponding GPU tests.
