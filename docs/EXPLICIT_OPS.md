# Explicit operation reference

Stateless forward and vector-Jacobian product (VJP) calls for applications that
manage saved tensors themselves. For ordinary autograd, use the
[semantic operations](OPS.md).


## Contents

- [Calling rules](#calling-rules)
- [Import and calling convention](#import-and-calling-convention)
- [Tensor conventions](#tensor-conventions)
- [Quick index](#quick-index)
- [Embedding](#embedding)
- [Normalization](#normalization)
- [Activations and rotary position](#activations-and-rotary-position)
- [Attention and sequence operations](#attention-and-sequence-operations)
- [Mixture of experts](#mixture-of-experts)
- [Losses](#losses)
- [Chunked language-model losses](#chunked-language-model-losses)
- [Operations without explicit VJPs](#operations-without-explicit-vjps)
- [Adding an explicit operation](#adding-an-explicit-operation)

## Calling rules

Every module exports exactly two functions:

```text
forward(*primal_arguments) -> output_and_explicit_residuals
backward(output_cotangent, *primal_or_residual_arguments) -> input_vjps
```

The contract is deliberately flat:

- no opaque context or `saved` object is created;
- every backward dependency is a named argument;
- forward returns every operation-produced tensor needed by backward;
- the caller owns residual storage, sizing, lifetime, and transfer;
- backward returns gradient values and never accumulates into `.grad`;
- both calls run with autograd recording disabled, even when arguments have
  `requires_grad=True`;
- calls are functional: they do not mutate or alias caller arguments; and
- explicit backward is first-order only.

The surface allocates and returns fresh output tensors.

These entry points use the same implementation resolver and raw implementation math
as the semantic surface; they are not a second copy of dispatch. Select an
exact implementation with `use_implementation` or a frozen manifest around
both calls. Apply-only native-Torch graphs are rejected by the explicit
surface because they rely on ordinary autograd rather than a raw VJP.

## Import and calling convention

Import the operation module, not its functions individually:

```python
from mlops.explicit import rms_norm

output, rstd = rms_norm.forward(x, weight, eps=1e-5)
grad_x, grad_weight = rms_norm.backward(
    grad_output,
    x,
    weight,
    rstd,
)
```

The semantic API keeps the residuals inside registered autograd:

```python
from mlops import rms_norm

output = rms_norm(x, weight)  # ordinary registered autograd
```

Models and blocks use the semantic API. Manual runtimes, isolated kernel
tests, and profilers may use the explicit API. Blocks and models remain
forward-only and intentionally expose no hand-authored backward methods.

## Tensor conventions

| Symbol | Meaning |
|---|---|
| `*` | arbitrary leading dimensions |
| `T` | flattened token count |
| `R` | flattened row count |
| `D` | model or feature width |
| `V` | vocabulary/embedding rows |
| `Hq`, `Hkv` | query and key/value heads |
| `Dh`, `Dv` | query/key and value head widths |
| `E`, `K`, `F` | experts, selected experts, expert hidden width |

Unless stated otherwise, floating-point primal inputs share a device and
storage dtype. Statistics such as `rstd`, `mean`, and attention LSE use FP32.
`lengths` is host metadata with positive entries summing to `T`. Integer IDs,
positions, routing indices, and sequence metadata are non-differentiable.

## Quick index

| Module | Forward public result | Forward residuals for backward | Backward result |
|---|---|---|---|
| `cross_entropy` | per-row FP32 losses | FP32 log-sum-exp | logits |
| `embedding` | embedding rows | none produced; retain token IDs | table gradient |
| `rms_norm` | normalized output | `rstd` | input, weight |
| `layer_norm` | normalized output | `mean`, `rstd` | input, weight, bias |
| `l2_norm` | normalized output | `rstd`; retain output | input |
| `gated_rms_norm` | gated output | `rstd` | input, gate, weight |
| `gelu` | activation | none produced; retain input | input |
| `swiglu` | activation | none produced; retain gate/up | gate, up |
| `packed_swiglu` | activation | none produced; retain packed input | packed input |
| `rope` | rotated tensor | none produced; retain positions | input |
| `partial_rope` | rotated tensor | none produced; retain positions | input |
| `flash_attention` | attention output | LSE and sequence metadata | Q, K, V |
| `dsa_attention` | selected-attention output | LSE; retain selected indices | Q, K, V |
| `sparse_latent_attention` | shared-latent attention output | output and LSE; retain selected indices | Q, combined KV |
| `kimi_delta_attention` | recurrent output | cumulative gates, Aqk, Akk and sequence metadata | Q, K, V, log decay, beta |
| `causal_conv_silu` | convolution output | caller retains cumulative sequence lengths | input, weight |
| `linear_attention` | recurrent output | FLA gate/matrix and sequence metadata | seven continuous inputs |
| `moe` | routed output and diagnostics | complete route/dispatch state | hidden, residual, router, expert weights |
| `head_loss` | scalar loss | seed-one hidden/head VJPs | scaled hidden/head VJPs |
| `lora_head_loss` | scalar loss | requested hidden/base/A/B seeds | scaled requested VJPs |

## Embedding

### `explicit.embedding`

**Signatures**

```text
forward(tokens: Tensor[*], weight: Tensor[V,D]) -> Tensor[* ,D]

backward(
    tokens: Tensor[*],
    grad_output: Tensor[* ,D],
    num_embeddings: int,
) -> Tensor[V,D]
```

`forward` performs table lookup. `backward` returns the deterministic dense
table gradient, including accumulation for repeated IDs. There is no gradient
for integer `tokens`; the caller supplies `weight.shape[0]` as
`num_embeddings`. Unlike the other modules' backward, this one takes `tokens`
before the cotangent.

## Normalization

### `explicit.rms_norm`

**Signatures**

```text
forward(
    x: Tensor[...,D],
    weight: Tensor[D],
    eps: float = 1e-5,
    *,
    zero_centered: bool = False,
) -> tuple[Tensor[...,D], Tensor[R]]

backward(
    grad_output: Tensor[...,D],
    x: Tensor[...,D],
    weight: Tensor[D],
    rstd: Tensor[R],
    *,
    zero_centered: bool = False,
) -> tuple[Tensor[...,D], Tensor[D]]
```

Forward returns `(output, rstd)`. `rstd` is FP32 and is the only
operation-produced residual. Backward returns `(grad_x, grad_weight)`, and
takes the `zero_centered` its forward was given.

### `explicit.layer_norm`

**Signatures**

```text
forward(
    x: Tensor[...,D],
    weight: Tensor[D],
    bias: Tensor[D] | None = None,
    eps: float = 1e-5,
) -> tuple[Tensor[...,D], Tensor[R], Tensor[R]]

backward(
    grad_output: Tensor[...,D],
    x: Tensor[...,D],
    weight: Tensor[D],
    mean: Tensor[R],
    rstd: Tensor[R],
) -> tuple[Tensor[...,D], Tensor[D], Tensor[D]]
```

Forward returns `(output, mean, rstd)` with FP32 statistics. Backward returns
`(grad_x, grad_weight, grad_bias)`. The affine-bias gradient is always
computed; discard it when forward used `bias=None`.

### `explicit.l2_norm`

**Signatures**

```text
forward(x: Tensor[...,D], eps: float = 1e-6)
    -> tuple[Tensor[...,D], Tensor[*]]

backward(
    grad_output: Tensor[...,D],
    output: Tensor[...,D],
    rstd: Tensor[*],
    eps: float = 1e-6,
) -> Tensor[...,D]
```

Forward returns the FLA normalized output and FP32 reciprocal norm. Backward
uses the saved normalized `output`, not the original input. This preserves the
documented FLA BF16 backward precision semantics.

### `explicit.gated_rms_norm`

**Signatures**

```text
forward(
    x: Tensor[...,D],
    gate: Tensor[...,D],
    weight: Tensor[D],
    eps: float = 1e-5,
    *, activation: str = "silu",
) -> tuple[Tensor[...,D], Tensor[R]]

backward(
    grad_output: Tensor[...,D],
    x: Tensor[...,D],
    gate: Tensor[...,D],
    weight: Tensor[D],
    rstd: Tensor[R],
    eps: float = 1e-5,
    *, activation: str = "silu",
) -> tuple[Tensor[...,D], Tensor[...,D], Tensor[D]]
```

The operation computes `silu(gate) * RMSNorm(x) * weight`, with
`sigmoid(gate)` under `activation="sigmoid"`. Backward returns
`(grad_x, grad_gate, grad_weight)`.


`activation` accepts `"silu"` or `"sigmoid"`; other values raise `ValueError`. Use the same choice in explicit forward and backward.

## Activations and rotary position

### `explicit.gelu`

```text
forward(x: Tensor[*]) -> Tensor[*]
backward(grad_output: Tensor[*], x: Tensor[*]) -> Tensor[*]
```

Applies tanh-approximate GELU and its explicit ATen VJP. Retain `x` for
backward.

### `explicit.swiglu`

```text
forward(
    gate: Tensor[...,F],
    up: Tensor[...,F],
    *, limit: float | None = None,
) -> Tensor[...,F]

backward(
    grad_output: Tensor[...,F],
    gate: Tensor[...,F],
    up: Tensor[...,F],
    *, limit: float | None = None,
) -> tuple[Tensor[...,F], Tensor[...,F]]
```

Returns `silu(gate) * up`; backward returns `(grad_gate, grad_up)`. Force
`builtin.swiglu.triton` or the ablation variant
`builtin.swiglu.triton_legacy` around both calls when selecting a variant.


`limit=None` preserves unclipped behavior. A finite positive `limit` clips gate values above the limit and up values to `[-limit, limit]`; derivatives include the clamp masks. The Triton path fuses this into its forward/backward kernels. Use the same limit in both explicit calls.

### `explicit.packed_swiglu`

```text
forward(packed: Tensor[...,2F], *, limit: float | None = None) -> Tensor[...,F]
backward(
    grad_output: Tensor[...,F],
    packed: Tensor[...,2F],
    *, limit: float | None = None,
) -> Tensor[...,2F]
```

The first half of `packed` is the gate and the second is the up projection.


`limit=None` preserves unclipped behavior. A finite positive `limit` clips gate values above the limit and up values to `[-limit, limit]`; derivatives include the clamp masks. The Triton path fuses this into its forward/backward kernels. Use the same limit in both explicit calls.

### `explicit.rope`

```text
forward(
    x: Tensor[T,H,Dh],
    positions: integer Tensor[T],
    base: float,
    cosine: Tensor[P,Dh],
    sine: Tensor[P,Dh],
) -> Tensor[T,H,Dh]

backward(
    grad_output: Tensor[T,H,Dh],
    positions: integer Tensor[T],
    base: float,
    cosine: Tensor[P,Dh],
    sine: Tensor[P,Dh],
) -> Tensor[T,H,Dh]
```

RoPE produces no saved activation. Backward applies the inverse rotation using
the same caller-owned `positions`, `base`, `cosine`, and `sine` values. Force
`builtin.rope.triton_table` or `builtin.rope.triton_analytic` around both calls
when selecting a variant.

### `explicit.partial_rope`

```text
forward(x, positions, base, rotary_dim, cosine, sine) -> Tensor[T,H,Dh]
backward(grad_output, positions, base, rotary_dim, cosine, sine) -> Tensor[T,H,Dh]
```

Only the leading `rotary_dim` channels are rotated. Remaining channels are
copied in forward and backward.

## Attention and sequence operations

### `explicit.flash_attention`

```text
forward(
    q: Tensor[T,Hq,Dh],
    k: Tensor[T,Hkv,Dh],
    v: Tensor[T,Hkv,Dh],
    cu_seqlens: int32 Tensor[L+1],
    max_seqlen: int,
    lengths: Sequence[int],
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> tuple[output, lse]

backward(
    grad_output, q, k, v, output, lse,
    cu_seqlens, max_seqlen, lengths,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> tuple[grad_q, grad_k, grad_v]
```

`output` has shape `[T,Hq,Dh]`; `lse` is FP32 `[Hq,T]`.
`cu_seqlens` contains caller-constructed packed boundaries and `lengths` the
same boundaries as host integers. Both calls raise `ValueError` unless the
lengths are positive and cover `T`, `cu_seqlens` is one-dimensional INT32 with
one more entry, `max_seqlen` is the longest length, and the inputs are
contiguous; unlike the semantic surface, no sequence may be empty. The explicit
surface accepts `builtin.flash_attention.aten`; apply-only
`native_torch.flash_attention` is rejected rather than replaying autograd. The
explicit backward does not read `deterministic_kernels`.

### `explicit.dsa_attention`

```text
forward(q, k, v, indices, lengths) -> tuple[output, lse]
backward(grad_output, q, k, v, indices, lse, lengths)
    -> tuple[grad_q, grad_k, grad_v]
```

`q` and `k` have shape `[T,Hq,Dqk]`, `v` has `[T,Hq,Dv]`, and `indices` has
`[T,K]`. Selected indices are discrete fixed inputs and receive no gradient.

### `explicit.kimi_delta_attention`

`forward(q, k, v, g, beta, cumulative, chunks)` returns
`(output, gates, aqk, akk)`. Retain the original inputs and all three
residuals for
`backward(dy, q, k, v, g, beta, gates, aqk, akk, cumulative, chunks)`,
which returns dQ, dK, dV, dG and dBeta. `gates` is the FP32 cumulative
decay; `aqk` and `akk` are provider-owned intra-chunk residuals. The native
cuLA recompute provider returns empty placeholders and rebuilds these matrices
in backward. Keep the same implementation override for both calls and do not
mix residuals from different providers. The semantic KDA constraints
also apply here. Calls do not build an autograd graph.

### `explicit.sparse_latent_attention`

`forward(q, kv, indices, *, scale)` returns `(output, lse)`.
`backward(dy, q, kv, indices, output, lse, *, scale)` returns dQ and dKV.
Retain inputs, output and FP32 softmax `lse`; pass the same `scale`
to both calls. The semantic tied-latent constraints also apply here.

With `q_rope` and `k_rope` supplied to both functions, `backward` returns
`(dq, dkv, dq_rope, dk_rope)`. Retain the same already-rotated positional inputs
from forward. Omitting both preserves the two-gradient return value.

### `explicit.causal_conv_silu`

```text
forward(
    x: Tensor[T,C],
    weight: Tensor[C,1,Wc],
    cumulative_lengths: Tensor[*],
    chunk_indices: Tensor[*] | None = None,
) -> Tensor[T,C]

backward(
    grad_output: Tensor[T,C],
    x: Tensor[T,C],
    weight: Tensor[C,1,Wc],
    cumulative_lengths: Tensor[*],
    chunk_indices: Tensor[*] | None = None,
) -> tuple[Tensor[T,C], Tensor[C,1,Wc]]
```

Convolution state resets at every packed boundary. Caller-owned cumulative
metadata may be an empty tensor for a single sequence and must be passed to
forward and backward unchanged. Optional caller-owned `chunk_indices` follows
the same contract and avoids reconstructing the chunk map inside provider
kernels.

### `explicit.linear_attention`

```text
forward(
    q, k, v, beta, a, a_log, dt_bias,
    cumulative, chunk_indices, *, scale=None,
) -> tuple[output, gate, matrix]

backward(
    grad_output, q, k, v, beta, a, a_log, dt_bias,
    gate, matrix, cumulative, chunk_indices, *, scale=None,
) -> tuple[grad_q, grad_k, grad_v, grad_beta, grad_a, grad_a_log, grad_dt_bias]
```

Q/K use `[T,Hk,Dk]`, V uses `[T,Hv,Dv]`, and `beta`/`a` use `[T,Hv]`.
`a_log` and `dt_bias` use `[Hv]`. `cumulative` and `chunk_indices` are
caller-owned round inputs and must be passed unchanged to backward. Forward's
FLA `gate` and `matrix` are explicit residuals and must remain immutable.
Omitting `scale` resolves it independently in both calls as `Dk**-0.5`.

## Mixture of experts

### `explicit.moe`

**Forward signature**

```text
forward(
    h2, residual, router_weight, w13_experts, w2_experts,
    *, top_k: int, routing_mode: str,
    router_bias=None, n_group: int = 1, topk_group: int = 1,
    routed_scaling: float = 1.0, lengths=None,
    weight_precision: str = "float32",
    swiglu_implementation: str = "builtin.packed_swiglu.triton",
) -> tuple[
    output, aux, counts, probability_sum, logits, route_weights,
    route_ids, order, offsets, slots, h13,
]
```

**Backward signature**

```text
backward(
    grad_output, grad_aux, grad_probability_sum,
    h2, router_weight, w13_experts, w2_experts,
    logits, route_weights, route_ids, order, offsets, slots, h13,
    *, top_k: int, routing_mode: str, lengths: Sequence[int],
    swiglu_implementation: str = "builtin.packed_swiglu.triton",
) -> tuple[grad_h2, grad_residual, grad_router, grad_w13, grad_w2]
```

`h2` and `residual` have `[...,D]`; `router_weight` is `[E,D]`;
`w13_experts` is `[E,2F,D]` (gate rows, then up rows); and `w2_experts` is
`[E,D,F]` -- every weight in `nn.Linear`'s `[out, in]` orientation.
Backward requires cotangents for the differentiable public `output`, `aux`, and
`probability_sum` roots. `counts` and discrete routing/dispatch metadata have
no VJP. Forward and backward must use the same `top_k`, `routing_mode`, packed
lengths, and builtin routed-SwiGLU implementation. The explicit entry points
take no `swiglu_limit`: their routed SwiGLU is unclamped.

This atomic operation covers the routed expert path only. Shared-expert
composition remains ordinary block/model computation and is not reproduced in
this explicit entry point.

## Losses

### `explicit.cross_entropy`

**Signatures**

```text
forward(
    logits: Tensor[R,V],
    targets: integer Tensor[R],
    weight: Tensor[V] | None = None,
    ignore_index: int = -100,
) -> tuple[Tensor[R], Tensor[R]]

backward(
    grad_losses: Tensor[R],
    logits: Tensor[R,V],
    targets: integer Tensor[R],
    logsumexp: Tensor[R],
    weight: Tensor[V] | None = None,
    ignore_index: int = -100,
) -> Tensor[R,V]
```

Forward returns `(losses, logsumexp)` before reduction. For production dtypes,
both are FP32. The caller retains `logits`, `targets`, optional class `weight`,
and `logsumexp`. Backward accepts an arbitrary per-row `grad_losses` and returns
only `grad_logits`; integer targets and class weights are non-differentiable.
Neither call records autograd or writes `.grad`.

The builtin Triton explicit path is unweighted. Passing `weight` selects the
native-Torch implementation unless an exact unsupported override was forced.

## Chunked language-model losses

The head operation uses a memory-bounded one-pass contract: forward creates
each chunk-local logits tensor once, immediately forms its cross-entropy
gradient, and returns seed-one parameter/input VJPs. Its backward entry point
only scales those immutable seeds by the caller-supplied scalar loss
cotangent.

### `explicit.head_loss`

```text
forward(
    hidden: Tensor[...,D], head_weight: Tensor[V,D],
    targets: integer Tensor[*],
    *, chunk_size: int | None = None,
    valid_rows: int | None = None,
) -> tuple[loss, grad_hidden_seed, grad_head_seed]

backward(
    grad_loss, grad_hidden_seed, grad_head_weight_seed,
) -> tuple[grad_hidden, grad_head_weight]
```

`loss` is an FP32 scalar. `grad_hidden_seed` has the original hidden shape;
the head seed matches its parameter. Seeds must remain immutable so the
same forward result can support repeated VJPs.


### `explicit.lora_head_loss`

```text
forward(
    hidden, head_weight, lora_a, lora_b, targets,
    *, scale=1.0, chunk_size=None, valid_rows=None, reduction="mean",
    weight_grad_dtype=None, need_hidden_grad=True, need_head_grad=False,
    need_lora_a_grad=True, need_lora_b_grad=True,
) -> tuple[loss, grad_hidden_seed, grad_head_seed, grad_lora_a_seed, grad_lora_b_seed]

backward(
    grad_loss, grad_hidden_seed, grad_head_seed, grad_lora_a_seed, grad_lora_b_seed,
) -> tuple[grad_hidden, grad_head, grad_lora_a, grad_lora_b]
```

Gradient requests are explicit and independent of input `requires_grad`.
An omitted seed and its backward result are `None`. The default requests
hidden/A/B gradients and omits the frozen base gradient. Hidden gradients have
the hidden dtype; parameter seeds use `weight_grad_dtype` or compute dtype.
Seeds are immutable, and calls neither build an autograd graph nor write
input `.grad`. Factor storage dtype can differ from the compute/seed dtype;
the explicit caller handles any final cast or accumulation.

## Operations without explicit VJPs

Not every model-facing helper is an atomic differentiable operation:

| Semantic helper | Reason there is no explicit module |
|---|---|
| `linear_mixer` | graph-layout/control-flow selection around a caller function |
| `dsa_indexer_key_norm`, `index_scores`, `sparse_mla`, `sigmoid_topk`, `kda_decay`, `mhc_normalize_streams`, `mhc_coefficients` | ordinary differentiable PyTorch compositions |
| `causal_conv_silu_qkv`, `mhc_pre`, `mhc_combine` | semantic-only: no implementation registers explicit forward/backward, and ordinary autograd differentiates them |
| `topk_indices`, `selection_mask`, `pooled_topk` | discrete selection with no mathematical VJP |
| `build_rope_tables`, `prepare_packed_sequence_metadata` | construction helpers with no differentiable outputs |
| implementation/diagnostic context managers | execution policy rather than tensor operations |

The package does not hand-author composite block or model backward functions.
Those remain forward-only PyTorch compositions; an external AOT lowering layer
may generate stage graph pairs when needed.

## Adding an explicit operation

When an operation gains its first implementation with explicit `forward` and
`backward`:

1. Add `mlops/explicit/<operation>.py` in the same change.
2. Export exactly `forward` and `backward` from that module.
3. Give both functions complete typed signatures and docstrings.
4. Wrap the actual call in `torch.no_grad()`; do not invoke
   `torch.autograd.grad`, `Tensor.backward`, or mutate `.grad`.
5. Return operation-produced residuals directly from forward and name every
   backward dependency explicitly.
6. Return gradient values without accumulation or dtype-policy side effects.
7. Document shapes, dtypes, non-differentiable inputs, residual immutability,
    implementation constraints, and ordered returns here.
8. Add explicit-forward and arbitrary-cotangent VJP parity against the
   registered semantic operation.
9. Test that outputs have no `grad_fn`, no input `.grad` is populated, and a
   repeated backward leaves every supplied argument unchanged.

The coverage test fails unless the explicit modules are exactly the
operations with such an implementation.
