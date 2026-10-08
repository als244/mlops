# LoRA modules

LoRA changes selected projections during model construction. Forward definitions
stay ordinary PyTorch code. Construct the optimizer after conversion.

```python
import torch
from mlops.lora import LoRAConfig, apply_lora, parameter_report

model = apply_lora(
    model,
    LoRAConfig(rank=32, alpha=32, factor_dtype=torch.float32),
    targets=["blocks.*.attn.wq", "blocks.*.attn.wv"],
)
print(parameter_report(model))
optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
```

## Configuration and trainability

`targets` contains module-path globs. Ordinary `nn.Linear` modules become
`LoRALinear`; unsupported targets or unmatched globs raise an error.
`trainable_base` contains parameter-path globs to keep fully trainable, such as
`["embed.weight"]`. All other original parameters freeze. The returned model is
the same object unless the root itself is selected with `targets=[""]`.
Conversion preserves original Parameter identities, names, values, aliases and
training mode. It adds `lora_a` and `lora_b` state-dict entries.

Each linear computes `x @ W.T + (alpha/rank) * (x @ A.T) @ B.T`.
A is initialized randomly and B to zero, so conversion initially preserves the
base model output. No full-sized `B @ A` is formed. Rank and alpha must be
positive; factor storage dtype is configurable and defaults to FP32. Compute
uses the activation dtype. Dropout and merged inference weights are not
implemented. Save/load the ordinary full state dict after applying the same
configuration. Factor-only checkpoint export is not part of this interface.

## Language-model heads

`mlops.modules.LanguageModelHead` has ordinary logits `forward` and bounded
`loss(hidden, targets, ...)` methods. Selecting it produces `LoRAHead`; both
methods then use the low-rank update. The loss calls
[`lora_head_loss`](OPS.md#lora_head_loss), omits frozen base-weight gradients,
and bounds logits workspace by processing rows in chunks. Directly reading a
head's `.weight` bypasses its factors; call the module or its `.loss` method.
A pure PyTorch head built from `nn.Linear` uses `LoRALinear` and its caller's
ordinary cross-entropy loss.

## Expert projections

Architecture recipes can pass `converters={ModuleType: factory}` to
`apply_lora`. A factory receives `(existing_module, config)` and returns a
replacement preserving the architecture's call signature. It must not modify
base parameters during construction. This keeps architecture choices out of
generic conversion and execution code.

Local grouped expert LoRA uses independent factors per expert, a joint gate/up
input factor and a separate down pair. For E experts, width D, hidden width H
and rank R, it adds `E * R * (2D + 3H)` parameters to `3E*D*H` base parameters.
The GPU implementation groups expert GEMMs; it neither launches a Python GEMM
loop per expert nor reconstructs full expert weights. Frozen base projections
compute input gradients but omit weight-gradient GEMMs and outputs. Routing,
activation and factor gradients remain necessary.

The separate [expert-parallel modules](EXPERT_PARALLEL.md) provide
`QuackMoELoRA` and `TEMoELoRA` with their own communication and factor-precision
configuration. They are not converted automatically by `apply_lora`.

## Memory and performance

LoRA removes frozen-parameter gradients and optimizer states. It does not remove
the frozen weights or the input-gradient matrix multiplies through them. It
also adds low-rank projections and their activations. For BF16 base weights,
FP32 factors/gradients and FP32 AdamW moments, with no master parameters, the
weight, gradient and optimizer tensor accounting is approximately:

- Full training: `14 * P` bytes (2 weight + 4 gradient + 8 moments).
- LoRA: `2 * P + 16 * L` bytes, where L is the factor count.

This counts all parameter gradients before a planner shortens their lifetimes;
it is not a measured simultaneous host or device peak. It excludes buffers,
temporary activations, allocator reserves, and scalar optimizer counters.
Saved/recomputed task workspaces and host RSS must still be measured. A large trainable-count reduction is not a proportional reduction
in forward compute or total task inputs. Optimizer parameter filtering and
frozen-weight gradient elision are tested separately.
