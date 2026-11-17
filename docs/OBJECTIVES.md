# Token objectives

Use `head_objective` to train a language-model head with bounded temporary
memory. The head projects a token chunk once, computes its loss and gradient
contributions, then releases its logits. Common objectives have fused CUDA
kernels; the same implementation handles full training and low-rank adapters.
See the [API reference](OPS.md#head_objective) for shapes and constraints.

## Cross entropy and SFT

```python
from mlops import head_objective

loss, logprobs = head_objective(
    hidden, head_weight, target_tokens,
    objective="cross_entropy",
    inputs=(loss_mask, weights),
)
loss.backward()
```

`loss_mask` is true at supervised positions and false at padding or ignored
prompt positions. For pretraining, supervise every real target; for SFT,
supervise the chosen response tokens. The loss is the sum of
`-weights * target_logprobs` at those positions. Choose weights or an outer loss
scale for the normalization your training loop needs. Returned logprobs are
detached diagnostics; training gradients come from the scalar loss.

Sampled distillation uses teacher-generated target tokens with the same CE
operation. This does not compute a full-vocabulary teacher/student KL divergence.

## Clipped policy loss

```python
loss, logprobs = head_objective(
    hidden, head_weight, sampled_tokens,
    objective="clipped_policy",
    inputs=(response_mask, weights, old_logprobs, advantages),
    controls=(clip_ratio,),
)
loss.backward()
```

The objective sums the negative PPO clipped surrogate, weighted over valid
positions. `clip_ratio` is a scalar device tensor. The caller supplies detached
old-policy logprobs and advantages; PPO and GRPO differ in how those advantages
are produced, not in this head kernel. Value-function losses and rollout/group
statistics stay outside the head operation.

To include the sampled k3 reference penalty, append `reference_logprobs` to
`inputs` and a scalar `reference_beta` tensor to `controls`. Per token, the added
term is `weight * beta * (exp(reference - current) - 1 - (reference - current))`.
A zero beta neutralizes the unused exponential before evaluation. The reference
policy is caller-owned; there is no hidden second model or forward pass.

## Frozen and low-rank heads

Pass `lora_a`, `lora_b` and `scale` to the same function. Set
`head_weight.requires_grad_(False)` to freeze the base. It then has no saved
weight gradient, while hidden states and trainable adapters still receive their
gradients. With all inputs frozen or under `torch.no_grad()`, the operation
computes only loss/logprobs. `LanguageModelHead.objective(...)` and
`LoRAHead.objective(...)` bind these parameters for model code.

## Precision and memory

The head follows the same `weight_gradients_at(dtype)` dispatch setting as other
parameterized operations. It saves parameter gradients in that dtype and hidden
gradients in the hidden tensor's dtype. Master weights and optimizer moments
belong to the optimizer, outside this operation.

For `T` tokens, hidden width `D` and vocabulary `V`, a fully trainable head saves
`V * D` weight-gradient elements and `T * D` hidden-gradient elements. Logits and
their gradients occupy only a token chunk at a time. Freezing the head removes
the `V * D` gradient buffer; LoRA adds gradients for its trainable factors.

Ordinary eager backward creates scaled gradient outputs while the saved
gradients remain alive, so both copies can contribute to peak memory. The
scaling uses ordinary tensor operations that a compiler can eliminate for a unit
scale or fuse with downstream accumulation. Saved gradients stay immutable so
repeated backward calls remain correct. No logits are recomputed in backward.

## Formulas on existing logprobs

For models that already produce differentiable target logprobs, use:

```text
mlops.objectives.cross_entropy(logprobs, valid, weights) -> Tensor
mlops.objectives.clipped_policy(
    logprobs, valid, weights, old_logprobs, advantages, clip_ratio,
    *, reference_logprobs=None, reference_beta=None,
) -> Tensor
```

Both return an FP32 sum with ordinary PyTorch autograd and no hidden
normalization. These are formulas on target logprobs;
[`mlops.cross_entropy`](OPS.md#cross_entropy) accepts vocabulary logits and
target IDs. The formulas also provide an independent autograd reference for
the fused head's manually calculated derivatives.

## Custom objective

1. Put a pure function in an installed Python module, for example
   `my_objectives.py`. It receives flat logprobs, a sequence of flat per-token
   inputs, and a sequence of scalar controls.
2. Return the scalar sum and its derivative with respect to each logprob.
   The sum must be separable across rows; compute whole-batch/group statistics
   before calling the head and pass them as detached inputs.
3. Compare its supplied derivative with ordinary PyTorch autograd on small
   cases, including masks and changed controls.
4. Pass that function as `objective`. A compiler/artifact integration must pin
   its source and validate its workspace behavior. To optimize a commonly used
   formula, add a fused implementation with the same semantics.

```python
import torch

def weighted_nll(logprobs, inputs, controls):
    valid, weights = inputs
    weight = torch.where(valid, weights.float(), 0.0)
    current = torch.where(valid, logprobs.float(), 0.0)
    return -(weight * current).sum(), -weight
```

This advanced function executes once per chunk and may launch several kernels.
The named built-ins above use the dedicated fused path. Custom functions cannot
return differentiable per-token outputs or receive gradients for their data and
controls through this operation; use an ordinary differentiable composition
when that is required.
