# Operations API

This is the complete operation reference for the standalone `mlops` package.
Public semantic calls are documented separately from implementation control,
contributor helpers, explicit forward/VJP entrypoints, and private dispatcher
boundaries.

## Contents

1. [API scope](#api-scope)
2. [Reference-entry format](#reference-entry-format)
3. [Tensor conventions](#tensor-conventions)
4. [Quick index](#quick-index)
5. [Embedding](#embedding)
6. [Normalization and activations](#normalization-and-activations)
7. [Rotary position operations](#rotary-position-operations)
8. [Dense and latent attention](#dense-and-latent-attention)
9. [DSA indexing and selected attention](#dsa-indexing-and-selected-attention)
10. [Hybrid linear-attention operations](#hybrid-linear-attention-operations)
11. [Mixture of experts](#mixture-of-experts)
12. [Language-model epilogues](#language-model-epilogues)
13. [Implementation-control API](#implementation-control-api)
14. [Private dispatcher boundary](#private-dispatcher-boundary)
15. [Operation-package example](#operation-package-example)
16. [Explicit operations](#explicit-operations)

## API scope

The stable model-facing import surface is:

```python
from mlops import (
    causal_conv_silu,
    cross_entropy,
    dsa_attention,
    dsa_indexer_key_norm,
    dsa_selection_override,
    embedding,
    flash_attention,
    gated_rms_norm,
    gelu,
    head_loss,
    index_scores,
    l2_norm,
    layer_norm,
    linear_attention,
    linear_mixer,
    mla_attention,
    moe,
    packed_swiglu,
    partial_rope,
    prepare_packed_sequence_metadata,
    rms_norm,
    rope,
    selection_mask,
    swiglu,
    topk_indices,
)
```

Semantic functions are the API used by models and blocks. Underscore-prefixed
dispatcher functions are private implementation boundaries. They may return
extra residuals required by registered autograd; model code must not call them.

### Stability labels

| Label | Meaning |
|---|---|
| **Public** | exported from `mlops`; suitable for models and blocks |
| **Contributor** | importable from its defining module; useful for implementing another semantic operation |
| **Private** | underscore-prefixed dispatcher/backend detail; no source compatibility promise |
| **Diagnostic** | public for verification or ablation, not intended in a frozen training artifact |

## Reference-entry format

Every public operation entry uses the same fields:

1. **Signature** — callable surface.
2. **Purpose** — semantic behavior.
3. **Parameters** — types, shapes, and meanings.
4. **Returns** — public return values and shapes.
5. **Autograd and effects** — gradients, saved state, mutation, and aliasing.
6. **Constraints and exceptions** — required relationships and expected errors.
7. **Implementations** — exact registered implementation IDs.
8. **Example** — minimal canonical use.

Unless an entry says otherwise, operations are first-order differentiable,
functional (`mutates_args=()` at registered boundaries), and return fresh
tensors that do not alias caller inputs.

## Tensor conventions

### Shape symbols

| Symbol | Meaning |
|---|---|
| `*` | arbitrary leading dimensions |
| `B` | rectangular batch size |
| `S` | tokens per rectangular row |
| `T` | flattened token count, normally `B × S` or `sum(lengths)` |
| `L` | number of packed sequences |
| `R` | flattened row count for a general `[..., width]` tensor |
| `D` | model/hidden width |
| `V` | vocabulary or embedding row count |
| `Hq` | query head count |
| `Hkv` | key/value head count |
| `Hi` | DSA indexer head count |
| `Dh` | dense-attention head width |
| `Dqk` | MLA/DSA query-key head width |
| `Dv` | value head width |
| `Di` | DSA indexer head width |
| `F` | dense or routed-expert hidden width |
| `E` | routed expert count |
| `K` | selected experts or DSA keys per token |
| `C` | causal-convolution channel count |
| `Wc` | causal-convolution kernel width |

Floating tensors normally share one device and compute dtype. Explicitly
identified statistics, logits reductions, routing probabilities, scores, and
losses use FP32. Integer metadata is non-differentiable.

### Packed sequences

`lengths` is a host `Sequence[int]` of positive segment lengths whose sum is
`T`. Attention, convolution, and recurrent operations reset at each boundary
and must not mix segments. Most semantic functions normalize it to a tuple or
dispatcher `list[int]`; it is specialized graph metadata rather than a device
tensor input. Gated linear attention additionally receives caller-owned device
metadata prepared once per packed round by
`prepare_packed_sequence_metadata`; providers never derive host counts from
CUDA tensors or retain metadata caches.

### Mutation, aliasing, and higher-order gradients

All registered model-facing differentiable operations currently declare
`mutates_args=()`. Private kernels may fill outputs allocated inside the
operation, but no caller input is written. Differentiable `*_fwd` operations
register fake/meta behavior and a VJP. Opaque VJP work uses a separate
`*_bwd` operation. Non-differentiable optimizer update effects use separate,
truthfully mutable targets documented in [OPTIMIZERS.md](OPTIMIZERS.md).

The package supports first-order training. Backward operations do not register
their own VJPs. Tensors saved by `register_autograd` are immutable; in
particular, head VJPs scale precomputed gradients out-of-place so retained-graph
replay is correct.

## Quick index

| Operation | Category | Public output | Default implementation |
|---|---|---|---|
| `embedding` | lookup | `[* ,D]` | `builtin.embedding.deterministic` |
| `rms_norm` | normalization | `[...,D]` | `builtin.rms_norm.triton` |
| `layer_norm` | normalization | `[...,D]` | `builtin.layer_norm.triton` |
| `l2_norm` | normalization | `[...,D]` | `fla.l2_norm` |
| `gated_rms_norm` | normalization/gating | `[...,D]` | `fla.gated_rms_norm` |
| `swiglu`, `packed_swiglu` | activation | `[...,F]` | `builtin.*.triton` |
| `gelu` | activation | shape of input | `builtin.gelu.aten_explicit_backward` |
| `rope`, `partial_rope` | position | shape of input | `builtin.*.triton` |
| `flash_attention` | dense attention | `[T,Hq,Dh]` | `builtin.flash_attention.aten` |
| `mla_attention` | latent attention | `[T,Hq,Dv]` | `builtin.mla_attention.flash` |
| `index_scores`, `topk_indices` | DSA selection | `[T,T]`, `[T,K]` | `builtin.index_scores.einsum` / not dispatched |
| `dsa_attention` | selected attention | `[T,Hq,Dv]` | `builtin.dsa_attention.sparse` |
| `causal_conv_silu` | hybrid mixer | shape of input | `fla.causal_conv_silu` |
| `linear_attention` | hybrid mixer | shape of V | `fla.linear_attention.gated_delta_rule` |
| `linear_mixer` | graph-layout policy | shape of hidden | `builtin.linear_mixer.packed` |
| `moe` | routed experts | output + diagnostics | `builtin.moe.grouped_gemm_composed` |
| `cross_entropy` | training loss | FP32 `[R]` or scalar | `builtin.cross_entropy.triton` |
| `head_loss` | bounded-logits training epilogue | FP32 scalar | `builtin.head_loss.chunked` |

## Embedding

### `embedding`

**Status:** Public

**Signature**

```python
embedding(tokens: Tensor, weight: Tensor) -> Tensor
```

**Purpose**

Looks up rows in an embedding table. The optimized backward performs a
deterministic sorted accumulation of repeated token IDs.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `tokens` | integer `Tensor[*]` | table-row indices |
| `weight` | `Tensor[V,D]` | embedding table |

**Returns**

`Tensor[* ,D]` on `weight.device` with `weight.dtype`.

**Autograd and effects**

Only `weight` is differentiable. Backward returns
`grad_weight[V,D]`; `tokens` is saved as non-differentiable metadata. No
input is mutated or aliased.

**Constraints and exceptions**

`tokens` must contain valid row indices. PyTorch raises the underlying index
error for values outside `[0,V)`.

**Implementations**

`builtin.embedding.deterministic` (default) or `native_torch.embedding`.

**Example**

```python
hidden = embedding(token_ids, model.embed.weight)  # [B,S,D]
```

## Normalization and activations

### `rms_norm`

**Status:** Public

**Signature**

```python
rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor
```

**Purpose**

Applies RMS normalization over the final dimension followed by an affine gain.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | `Tensor[...,D]` | rows to normalize |
| `weight` | `Tensor[D]` | learned gain |
| `eps` | `float` | positive squared-RMS stabilizer |

**Returns**

`Tensor[...,D]` with the shape and dtype of `x`.

**Autograd and effects**

The optimized forward additionally exposes a private FP32 `rstd[R]` residual
and saves `x`, `weight`, and `rstd`. Backward returns gradients for
`x` and `weight`. No input is mutated.

**Constraints and exceptions**

`weight.numel()` must equal `x.shape[-1]`; `x` must have a nonempty final
dimension. Backend shape errors are raised for incompatible inputs.

**Implementations**

`builtin.rms_norm.triton` (default), `liger.rms_norm`, or
`native_torch.rms_norm`.

**Example**

```python
normalized = rms_norm(hidden, norm.weight, eps=1e-5)
```

### `layer_norm`

**Status:** Public

**Signature**

```python
layer_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor
```

**Purpose**

Applies final-dimension LayerNorm with optional affine bias.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | `Tensor[...,D]` | rows to normalize |
| `weight` | `Tensor[D]` | affine gain |
| `bias` | `Optional[Tensor[D]]` | optional affine bias |
| `eps` | `float` | variance stabilizer |

**Returns**

`Tensor[...,D]` with `x.dtype`.

**Autograd and effects**

The registered forward privately returns FP32 `mean[R]` and `rstd[R]`.
Backward returns `grad_x`, `grad_weight`, and `grad_bias` when bias is
present. No input is mutated.

**Constraints and exceptions**

`weight` and optional `bias` must match `x.shape[-1]`.

**Implementations**

`builtin.layer_norm.triton` (default) or `native_torch.layer_norm`.

**Example**

```python
normalized = layer_norm(hidden, weight, bias, eps=1e-5)
```

### `l2_norm`

**Status:** Public

**Signature**

```python
l2_norm(x: Tensor, eps: float = 1e-6) -> Tensor
```

**Purpose**

L2-normalizes every final-dimension vector using an FP32 norm reduction.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | `Tensor[...,D]` | vectors to normalize |
| `eps` | `float` | squared-norm stabilizer |

**Returns**

`Tensor[...,D]` with the shape and dtype of `x`.

**Autograd and effects**

The FLA forward privately exposes FP32 `rstd[x.shape[:-1]]` and saves the
normalized output plus `rstd`. Its first-order VJP follows FLA's production
precision; `native_torch.l2_norm` is the exact expression oracle.

**Constraints and exceptions**

The final dimension must be nonempty. No independent output dtype is accepted.

**Implementations**

`fla.l2_norm` (default) or `native_torch.l2_norm`.

**Example**

```python
q = l2_norm(q)  # [T,H,D]
```

### `gated_rms_norm`

**Status:** Public

**Signature**

```python
gated_rms_norm(
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    eps: float = 1e-5,
) -> Tensor
```

**Purpose**

Computes the Qwen-hybrid gated normalization equivalent to
`silu(gate) * RMSNorm(x) * weight`, using implementation-specific fused ordering.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | `Tensor[...,D]` | values to normalize |
| `gate` | `Tensor[...,D]` | elementwise SiLU gate |
| `weight` | `Tensor[D]` | learned gain |
| `eps` | `float` | RMS stabilizer |

**Returns**

`Tensor[...,D]` with the shape and dtype of `x`.

**Autograd and effects**

The optimized path saves `x`, `gate`, `weight`, and private FP32
`rstd[R]`; backward returns gradients for all three tensor inputs.

**Constraints and exceptions**

`x.shape == gate.shape` and `weight.shape == (D,)`.

**Implementations**

`fla.gated_rms_norm` (default) or `native_torch.gated_rms_norm`.

**Example**

```python
output = gated_rms_norm(values, output_gate, norm.weight)
```

### `swiglu`

**Status:** Public

**Signature**

```python
swiglu(gate: Tensor, up: Tensor) -> Tensor
```

**Purpose**

Computes `silu(gate) * up` elementwise.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `gate` | `Tensor[...,F]` | gate projection |
| `up` | `Tensor[...,F]` | value projection |

**Returns**

`Tensor[...,F]` matching the inputs.

**Autograd and effects**

Saves `gate` and `up`; backward returns separate gradients. The default
Triton VJP is the corrected implementation. `triton_legacy` retains the old
backward only for historical ablation.

**Constraints and exceptions**

`gate.shape` must equal `up.shape`; both tensors must be compatible in
device and dtype.

**Implementations**

`builtin.swiglu.triton` (default), `builtin.swiglu.triton_legacy`, or
`native_torch.swiglu`.

**Example**

```python
activated = swiglu(gate_projection, up_projection)
```

### `packed_swiglu`

**Status:** Public

**Signature**

```python
packed_swiglu(packed: Tensor) -> Tensor
```

**Purpose**

Splits one packed projection into equal gate/up halves and applies SwiGLU.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `packed` | `Tensor[...,2F]` | gate channels followed by up channels |

**Returns**

`Tensor[...,F]`.

**Autograd and effects**

Saves `packed`; backward returns one `grad_packed[...,2F]`. Provider
selection is shared with `swiglu`.

**Constraints and exceptions**

The final width must be even.

**Implementations**

`builtin.packed_swiglu.triton` (default),
`builtin.packed_swiglu.triton_legacy`, or `native_torch.packed_swiglu`.

**Example**

```python
activated = packed_swiglu(project(hidden))  # project width is 2F
```

### `gelu`

**Status:** Public

**Signature**

```python
gelu(x: Tensor) -> Tensor
```

**Purpose**

Applies tanh-approximate GELU without changing shape.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | floating `Tensor[*]` | activation input |

**Returns**

A tensor matching `x`.

**Autograd and effects**

The default boundary uses the native forward expression with an explicit
`aten.gelu_backward` dispatcher operation. It saves `x`.

**Constraints and exceptions**

`x` must use a dtype supported by PyTorch GELU on its device.

**Implementations**

`builtin.gelu.aten_explicit_backward` (default) or `native_torch.gelu`.

**Example**

```python
hidden = gelu(input_projection)
```

## Rotary position operations

### `build_rope_tables`

```python
build_rope_tables(
    rows: int,
    rotary_dim: int,
    base: float,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]
```

Builds reusable FP32 `cosine` and `sine` tensors of shape
`[rows, rotary_dim]`. These are ordinary caller-owned tensors, not operation
workspace. A module may store them as non-persistent buffers.

### `rope`

**Status:** Public

**Signature**

```python
rope(
    x: Tensor,
    positions: Tensor,
    base: float,
    cosine: Tensor,
    sine: Tensor,
) -> Tensor
```

**Purpose**

Applies full rotary position encoding to the final head dimension.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | `Tensor[...,H,Dh]` | head vectors to rotate |
| `positions` | integer `Tensor[T]` | flattened token positions |
| `base` | `float` | rotary frequency base |
| `cosine` | FP32 `Tensor[P,Dh]` | reusable cosine table input |
| `sine` | FP32 `Tensor[P,Dh]` | reusable sine table input |

**Returns**

A fresh tensor matching `x`.

**Autograd and effects**

No floating activation produced by the operation is saved. Table-backed
backward retains ordinary references to `positions`, `cosine`, and `sine`.

**Constraints and exceptions**

`Dh` must be even. The flattened token extent represented by `x` must agree
with `positions.numel()`. Optimized paths explicitly materialize
non-contiguous channel slices before launching.

The default implementation is `builtin.rope.triton_table`; alternatives are
`builtin.rope.triton_analytic` and `native_torch.rope`. No implementation or
registry caches table tensors.

**Example**

```python
cosine, sine = build_rope_tables(4096, 128, 500_000.0, query.device)
query = rope(query, positions, 500_000.0, cosine, sine)
```

### `partial_rope`

**Status:** Public

**Signature**

```python
partial_rope(
    x: Tensor,
    positions: Tensor,
    base: float,
    rotary_dim: int,
    cosine: Tensor,
    sine: Tensor,
) -> Tensor
```

**Purpose**

Rotates only the leading `rotary_dim` channels and copies the remaining head
channels unchanged.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | `Tensor[...,H,Dh]` | head vectors |
| `positions` | integer `Tensor[T]` | flattened positions |
| `base` | `float` | rotary frequency base |
| `rotary_dim` | `int` | number of leading channels to rotate |
| `cosine` | FP32 `Tensor[P,rotary_dim]` | reusable cosine table input |
| `sine` | FP32 `Tensor[P,rotary_dim]` | reusable sine table input |

**Returns**

A fresh tensor matching `x`.

**Autograd and effects**

Saves positions and scalar metadata only. Backward inverse-rotates the leading
gradient channels and passes the tail through.

**Constraints and exceptions**

`0 <= rotary_dim <= Dh`; a nonzero rotary width must be even. Token metadata
must match the flattened token extent.

**Implementations**

`builtin.partial_rope.triton` (default) or `native_torch.partial_rope`.

**Example**

```python
query = partial_rope(
    query, positions, cfg.rope_base, cfg.rot_dim, cosine, sine
)
```

## Dense and latent attention

### `flash_attention`

**Status:** Public

**Signature**

```python
flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lengths: Sequence[int],
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> Tensor
```

**Purpose**

Runs block-diagonal variable-length MHA or GQA over flat token-major Q/K/V
tensors.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `q` | `Tensor[T,Hq,Dh]` | queries |
| `k` | `Tensor[T,Hkv,Dh]` | keys |
| `v` | `Tensor[T,Hkv,Dh]` | values |
| `lengths` | `Sequence[int]` | positive packed lengths summing to `T` |
| `causal` | `bool` | whether future keys are masked |
| `softmax_scale` | `Optional[float]` | explicit logit scale; `None` uses the backend default |

**Returns**

`Tensor[T,Hq,Dh]`.

**Autograd and effects**

The optimized registered path saves Q/K/V, output, cumulative sequence
metadata, and FP32 log-sum-exp when the native flash primitive supplies it.
The semantic boundary derives compact device offsets from `lengths` inside the
opaque forward and exposes them only as a private autograd residual; it retains
no tensor state. Backward returns gradients for Q/K/V. Inputs are made
contiguous when needed; caller tensors are not modified.

**Constraints and exceptions**

- Q/K/V must be rank three.
- `k.shape == v.shape`.
- Q and K head widths must match.
- `Hq % Hkv == 0`.
- `lengths` must be nonempty, positive, and sum to `T`.

Violations detected by the varlen boundary raise `ValueError`.

**Implementations**

`builtin.flash_attention.aten` (default) or
`native_torch.flash_attention`. The builtin implementation requires PyTorch's
native variable-length FlashAttention primitive and rejects unsupported calls;
it never silently falls back.

**Example**

```python
attended = flash_attention(q, k, v, lengths)  # [T,Hq,Dh]
```

### `mla_attention`

**Status:** Public

**Signature**

```python
mla_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lengths: Sequence[int],
) -> Tensor
```

**Purpose**

Runs attention on already assembled MLA query, key, and value heads.
Low-rank projections, normalization, and decoupled RoPE belong to the
surrounding `MLAProjections` block.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `q` | `Tensor[T,Hq,Dqk]` | assembled MLA queries |
| `k` | `Tensor[T,Hq,Dqk]` | assembled MLA keys |
| `v` | `Tensor[T,Hq,Dv]` | values |
| `lengths` | `Sequence[int]` | packed boundaries |

**Returns**

`Tensor[T,Hq,Dv]`.

**Autograd and effects**

Uses the selected dense-attention implementation. When flash attention requires
equal Q/K/V widths and `Dv < Dqk`, the operation zero-pads V internally and
slices the public output/gradient back to `Dv`.

**Constraints and exceptions**

`Dv <= Dqk`; otherwise `ValueError` is raised. Q/K token, head, and width
dimensions must match, and `lengths` must cover `T`.

**Implementations**

`builtin.mla_attention.flash` (default) or `native_torch.mla_attention`.

**Example**

```python
latent_output = mla_attention(query, key, value, lengths)
```

## DSA indexing and selected attention

### `dsa_indexer_key_norm`

**Status:** Public

**Signature**

```python
dsa_indexer_key_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
) -> Tensor
```

**Purpose**

Applies the DSA indexer's key LayerNorm with the selected implementation's cast and
reduction semantics used by index scoring.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | `Tensor[T,Di]` | index keys |
| `weight` | `Tensor[Di]` | LayerNorm gain |
| `bias` | `Tensor[Di]` | LayerNorm bias |
| `eps` | `float` | variance stabilizer |

**Returns**

`Tensor[T,Di]`.

**Autograd and effects**

Differentiable with respect to `x`, `weight`, and `bias`; no mutation.

**Constraints and exceptions**

Affine vectors must match `Di`.

**Implementations**

`builtin.dsa_indexer_key_norm.layer_norm` (default) or
`native_torch.dsa_indexer_key_norm`. Index scoring is selected independently.

**Example**

```python
key = dsa_indexer_key_norm(key_projection, gain, bias)
```

### `index_scores`

**Status:** Public

**Signature**

```python
index_scores(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    lengths: Sequence[int],
) -> Tensor
```

**Purpose**

Builds the FP32 causal, block-diagonal lightning-indexer score matrix.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `q` | `Tensor[T,Hi,Di]` | per-index-head queries |
| `k` | `Tensor[T,Di]` | shared keys |
| `weights` | `Tensor[T,Hi]` | per-token index-head weights |
| `lengths` | `Sequence[int]` | sequence boundaries |

**Returns**

FP32 `Tensor[T,T]`. Valid causal entries contain scores; future and
cross-sequence entries are `-inf`.

**Autograd and effects**

Ordinary PyTorch autograd differentiates Q/K/weights when called outside a
`no_grad` selection region. No input is mutated.

**Constraints and exceptions**

Token and index-head dimensions must agree. Lengths must cover the token axis.

**Implementations**

`builtin.index_scores.einsum` (default) or `native_torch.index_scores`.

**Example**

```python
scores = index_scores(index_q, index_k, head_weights, lengths)
```

### `topk_indices`

**Status:** Public

**Signature**

```python
topk_indices(
    scores: Tensor,
    lengths: Sequence[int],
    top_k: int,
) -> Tensor
```

**Purpose**

Selects up to `top_k` causal keys per token within each packed sequence and
returns global token indices.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `scores` | `Tensor[T,T]` | causal block-diagonal scores |
| `lengths` | `Sequence[int]` | sequence boundaries |
| `top_k` | positive `int` | rectangular selected width `K` |

**Returns**

Int32 `Tensor[T,K]`. Sequences shorter than `K` repeat an already selected
key to pad the representation; selection masks remove duplicate semantics.

**Autograd and effects**

Selection is discrete and non-differentiable. A diagnostic override may replace
the result outside compiler capture.

**Constraints and exceptions**

`scores` must cover each range in `lengths`; `top_k` must be positive.
Override tensors must be `[T,K]` and may not reference another sequence.

**Implementations**

This helper is not registry-dispatched. Its `scores` input may come from any
`index_scores` implementation.

**Example**

```python
indices = topk_indices(scores, lengths, top_k=2048)
```

### `selection_mask`

**Status:** Public helper

**Signature**

```python
selection_mask(
    indices: Tensor,
    lo: int,
    hi: int,
    row_lo: int,
    row_hi: int,
) -> Tensor
```

**Purpose**

Builds an additive selected-and-causal attention mask for a rectangular query
subrange against one sequence key range.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `indices` | integer `Tensor[T,K]` | global selected key IDs |
| `lo, hi` | `int` | half-open key range |
| `row_lo, row_hi` | `int` | half-open query-row range |

**Returns**

FP32 `Tensor[row_hi-row_lo, hi-lo]` containing `0` for selected causal
entries and `-inf` elsewhere.

**Autograd and effects**

Non-differentiable. The returned mask is fresh.

**Constraints and exceptions**

Ranges must be ordered and lie within the token domain represented by
`indices`.

**Implementations**

This helper is not registry-dispatched.

**Example**

```python
mask = selection_mask(indices, seq_start, seq_stop, seq_start, seq_stop)
```

### `dsa_selection_override`

**Status:** Public diagnostic context manager

**Signature**

```python
dsa_selection_override(
    decisions: Iterable[Tensor],
) -> ContextManager[None]
```

**Purpose**

Supplies predetermined DSA top-k decisions in call order so two implementations
can be compared without router/index-selection discontinuities.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `decisions` | iterable of integer `Tensor[T,K]` | detached selection tensors consumed one per `topk_indices` call |

**Returns**

A context manager yielding `None`.

**Autograd and effects**

Uses context-local Python diagnostic state. It is disabled while compiling and
must not be treated as part of a frozen graph artifact.

**Constraints and exceptions**

Entering code must consume exactly all supplied decisions. Too few, too many,
wrong-shaped, or cross-sequence decisions raise `RuntimeError` or
`ValueError`.

**Implementations**

This diagnostic context manager is not registry-dispatched.

**Example**

```python
with dsa_selection_override(reference_indices):
    candidate_loss = model.loss(tokens, targets, seq_lens=lengths)
```

### `dsa_attention`

**Status:** Public

**Signature**

```python
dsa_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    indices: Tensor,
    lengths: Sequence[int],
    *,
    reference_pad_value: bool = False,
) -> Tensor
```

**Purpose**

Runs selected sparse causal attention over explicit DSA key indices.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `q` | `Tensor[T,Hq,Dqk]` | queries |
| `k` | `Tensor[T,Hq,Dqk]` | keys |
| `v` | `Tensor[T,Hq,Dv]` | values |
| `indices` | integer `Tensor[T,K]` | selected global keys |
| `lengths` | `Sequence[int]` | packed boundaries |
| `reference_pad_value` | `bool` | on `native_torch.dsa_attention`, zero-pad V to Q/K width before SDPA |

**Returns**

`Tensor[T,Hq,Dv]`.

**Autograd and effects**

The sparse path privately returns FP32 log-sum-exp `[Hq,T]` and saves
Q/K/V, indices, and LSE. Backward returns gradients for Q/K/V only.

**Constraints and exceptions**

Q and K must have identical rank-three shapes; V must match their token/head
axes; lengths must cover `T`; selected IDs must remain in their sequence and
must not point to future tokens. Detected shape/length violations raise
`ValueError`.

**Implementations**

`builtin.dsa_attention.sparse` (default) or
`native_torch.dsa_attention`.

**Example**

```python
output = dsa_attention(query, key, value, indices, lengths)
```

## Hybrid linear-attention operations

### `causal_conv_silu`

**Status:** Public

**Signature**

```python
causal_conv_silu(
    x: Tensor,
    weight: Tensor,
    lengths: Sequence[int],
    cumulative: Tensor,
) -> Tensor
```

**Purpose**

Applies a packed depthwise causal one-dimensional convolution followed by SiLU.
Convolution history resets at every sequence boundary.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | `Tensor[...,C]`, flattened to `[T,C]` | channel-last input |
| `weight` | `Tensor[C,1,Wc]` | depthwise convolution kernels |
| `lengths` | `Sequence[int]` | boundaries summing to `T` |
| `cumulative` | INT64 `Tensor[L+1]`, or empty for one sequence | caller-owned cumulative boundaries |

**Returns**

A tensor matching `x`.

**Autograd and effects**

The FLA path saves `x`, `weight`, and the caller-owned cumulative sequence
metadata. Backward returns gradients for `x` and `weight`. The operation does
not construct or retain metadata.

**Constraints and exceptions**

`sum(lengths)` must equal the flattened row count or `ValueError` is
raised. Weight channels must match `C`.

**Implementations**

`fla.causal_conv_silu` (default) or `native_torch.causal_conv_silu`.

**Example**

```python
cumulative, chunks = prepare_packed_sequence_metadata(lengths, projected)
convolved = causal_conv_silu(projected, conv.weight, lengths, cumulative)
```

### `prepare_packed_sequence_metadata`

**Status:** Public preparation helper

**Signature**

```python
prepare_packed_sequence_metadata(
    lengths: Sequence[int],
    like: Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor]
```

**Purpose**

Prepares caller-owned cumulative boundaries and `(sequence, chunk)` indices
once per packed round. Its small registered preparation target keeps
call-derived values real and graph-visible during FakeTensor/AOT capture. It is
not provider-dispatched, has no differentiable outputs, and retains no state.

**Returns**

`(cumulative, chunk_indices)` as INT64 tensors on `like.device`. A single
sequence uses empty tensors with shapes `[0]` and `[0,2]`. Multi-sequence CUDA
results are copied nonblockingly from pinned host values.

**Constraints and exceptions**

Lengths and `chunk_size` must be positive. Callers pass the same returned
tensors to every causal-convolution and linear-attention layer in the round.
Providers consume but never retain or reconstruct them.

**Example**

```python
cumulative, chunks = prepare_packed_sequence_metadata(lengths, hidden)
```

### `linear_attention`

**Status:** Public

**Signature**

```python
linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    a: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    lengths: Sequence[int],
    cumulative: Tensor,
    chunk_indices: Tensor,
    *,
    scale: float | None = None,
) -> Tensor
```

**Purpose**

Runs the packed Gated DeltaNet recurrent/chunked delta-rule core.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `q` | `Tensor[T,Hk,Dk]` | normalized queries |
| `k` | `Tensor[T,Hk,Dk]` | normalized keys |
| `v` | `Tensor[T,Hv,Dv]` | values |
| `beta` | `Tensor[T,Hv]` | update gate |
| `a` | `Tensor[T,Hv]` | input-dependent decay term |
| `a_log` | `Tensor[Hv]` | learned log-decay parameter |
| `dt_bias` | `Tensor[Hv]` | learned decay bias |
| `lengths` | `Sequence[int]` | recurrent reset boundaries |
| `cumulative` | INT64 `Tensor[L+1]`, or empty for one sequence | caller-owned cumulative boundaries |
| `chunk_indices` | INT64 `Tensor[Nchunk,2]`, or empty for one sequence | caller-owned `(sequence, chunk)` map |
| `scale` | `Optional[float]` | query/key scale; default `Dk**-0.5` |

**Returns**

`Tensor[T,Hv,Dv]`, matching `v`.

**Autograd and effects**

The FLA boundary saves all differentiable inputs, the caller-owned metadata,
and private gate/matrix residuals. Backward returns seven tensor gradients.
`a_log` and `dt_bias` gradients are FP32 in the fake/FLA contract. Neither the
operation nor provider constructs or retains sequence metadata.

**Constraints and exceptions**

Token axes and value-head gating dimensions must agree. Lengths must be
positive and cover `T`. Multi-sequence calls require matching INT64 cumulative
and 64-token chunk-index tensors; single-sequence calls use empty sentinels.
Provider-specific unsupported shapes fail during implementation resolution.

**Implementations**

`fla.linear_attention.gated_delta_rule` (default) or
`native_torch.linear_attention` recurrent oracle.

**Example**

```python
cumulative, chunks = prepare_packed_sequence_metadata(lengths, q)
values = linear_attention(
    q, k, v, beta, a, A_log, dt_bias, lengths, cumulative, chunks
)
```

### `linear_mixer`

**Status:** Public policy helper

**Signature**

```python
linear_mixer(
    forward_segment: Callable[[Tensor, tuple[int, ...]], Tensor],
    hidden: Tensor,
    lengths: Sequence[int],
) -> Tensor
```

**Purpose**

Chooses whether a hybrid linear-mixer projection graph runs once on the packed
batch or separately per sequence. This is a graph-layout/fidelity ablation, not
another mathematical mixer.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `forward_segment` | callable | maps one hidden segment and its lengths to a same-shaped result |
| `hidden` | `Tensor[B,S,D]`; packed use is normally `[1,T,D]` | mixer input |
| `lengths` | `Sequence[int]` | segment lengths |

**Returns**

A tensor with the same shape as `hidden`.

**Autograd and effects**

Autograd follows the callback graph. Provider state is context-local and does
not mutate the callback or module.

**Constraints and exceptions**

The callback must preserve batch/sequence concatenation semantics. In
`per_sequence` mode, outputs must be concatenable along dimension 1.

**Implementations**

`builtin.linear_mixer.packed` (default) or
`native_torch.linear_mixer.per_sequence`.

**Example**

```python
output = linear_mixer(self._forward_segment, hidden, lengths)
```

## Mixture of experts

### `moe`

**Status:** Public

**Signature**

```python
moe(
    h2: Tensor,
    residual: Tensor,
    router_weight: Tensor,
    w13_experts: Tensor,
    w2_experts: Tensor,
    *,
    top_k: int,
    routing_mode: str,
    router_bias: Tensor | None = None,
    n_group: int = 1,
    topk_group: int = 1,
    routed_scaling: float = 1.0,
    lengths: Sequence[int] | None = None,
    shared_w13: Tensor | None = None,
    shared_w2: Tensor | None = None,
    shared_gate_weight: Tensor | None = None,
    shared_mode: str = "add",
) -> tuple[Tensor, Tensor, Tensor, Tensor]
```

**Purpose**

Owns router scoring, discrete route selection, grouped dispatch, expert
SwiGLU, shared-expert composition, residual combination, and load-balance
diagnostics.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `h2` | `Tensor[...,D]`, `R` flattened rows | normalized expert input |
| `residual` | same shape as `h2` | residual base |
| `router_weight` | `Tensor[D,E]` | router projection |
| `w13_experts` | `Tensor[E,D,2F]` | packed gate/up expert weights |
| `w2_experts` | `Tensor[E,F,D]` | expert down weights |
| `top_k` | positive `int` | routes selected per row |
| `routing_mode` | `str` | `softmax_then_topk`, `topk_then_softmax`, or `sigmoid_noaux_tc` |
| `router_bias` | `Optional[Tensor[E]]` | optional bias used for selection |
| `n_group` | positive `int` | expert routing group count |
| `topk_group` | positive `int` | groups retained before expert selection |
| `routed_scaling` | `float` | multiplier for selected route weights |
| `lengths` | `Optional[Sequence[int]]` | sequence-wise auxiliary-loss boundaries; omitted means one segment |
| `shared_w13` | `Optional[Tensor[D,2Fs]]` | optional shared gate/up projection |
| `shared_w2` | `Optional[Tensor[Fs,D]]` | optional shared down projection |
| `shared_gate_weight` | tensor compatible with `[D,1]` | required by sigmoid-gated shared mode |
| `shared_mode` | `Literal["add", "sigmoid"]` | shared-expert combination |

**Returns**

| Position | Type/shape | Meaning |
|---|---|---|
| 0 | tensor matching `residual` | residual plus routed/shared expert output |
| 1 | FP32 scalar | routing auxiliary term |
| 2 | integer `Tensor[E]` | selected assignment count per expert |
| 3 | FP32 `Tensor[E]` | summed router probability per expert |

**Autograd and effects**

The registered path privately saves router logits, route weights/IDs, sorted
assignment order, expert offsets, inverse slots, and packed pre-activation
expert projections. Backward returns gradients for `h2`, `residual`,
`router_weight`, `w13_experts`, and `w2_experts`. Router bias, counts,
probability diagnostics, and discrete route metadata do not receive gradients.
The operation itself does not update router-bias or round counters.

**Constraints and exceptions**

- `h2.shape == residual.shape`.
- Expert tensor dimensions must agree with `D`, `E`, and `F`.
- `1 <= top_k <= E`.
- Shared W13/W2 must be supplied together.
- `shared_mode` must be `add` or `sigmoid`.
- Sigmoid mode requires `shared_gate_weight`.
- Group counts and `lengths` must be compatible with routing mode.

Invalid shared-expert combinations raise `ValueError`; other incompatible
shapes or implementations raise their backend error.

**Implementations**

`builtin.moe.grouped_gemm_composed` (default), the single-region opaque control
`builtin.moe.grouped_gemm`, `scattermoe.moe`, or `native_torch.moe`. The
composed builtin keeps the same semantic call but exposes routing/dispatch/
`w13` separately from SwiGLU/`w2`/combine so AOT can rematerialize `h13`
without replaying the expert down projection. Routed expert activation remains
part of the selected MoE implementation; it does not consult the standalone
`swiglu` operation.

The manual explicit forward/backward API continues to resolve to the opaque
whole-operation implementation. Composition is a semantic compiler boundary,
not a second public residual ABI. Select the opaque semantic control explicitly
with `use_implementation("moe", "builtin.moe.grouped_gemm")`.

**Example**

```python
output, aux, counts, probability_sum = moe(
    normalized,
    residual,
    router_weight,
    w13,
    w2,
    top_k=2,
    routing_mode="topk_then_softmax",
)
```

## Language-model epilogues

### `cross_entropy`

**Status:** Public

**Signature**

```python
cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: Literal["none", "sum", "mean"] = "mean",
) -> Tensor
```

**Purpose**

Computes class-index cross entropy directly from logits. This is the reusable
logits-to-loss operation; it does not perform final normalization, vocabulary
projection, or chunked head execution.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `logits` | floating `Tensor[R,V]` | unnormalized class scores |
| `targets` | int32/int64 `Tensor[R]` | target class per row or `ignore_index` |
| `weight` | optional floating `Tensor[V]` | non-differentiable per-class loss weight |
| `ignore_index` | `int` | target value that contributes zero loss and gradient |
| `reduction` | `"none"`, `"sum"`, or `"mean"` | per-row, summed, or valid-weight mean result |

**Returns**

With `reduction="none"`, returns FP32 losses `[R]` for ordinary production
dtypes. `"sum"` and `"mean"` return an FP32 scalar. The native-Torch float64
development path preserves float64 for numerical gradcheck. Weighted mean uses
the sum of weights for non-ignored targets, matching PyTorch semantics.

**Autograd and effects**

Only `logits` is differentiable. The opaque builtin forward privately returns
FP32 log-sum-exp `[R]`, and registered autograd saves ordinary references to
`logits`, `targets`, and that residual. Backward accepts the per-row cotangent
produced by the reduction graph and returns `grad_logits[R,V]`. Inputs and
cotangents are not mutated; first-order retained-graph replay is supported.

**Constraints and exceptions**

`logits` must be rank two with nonempty `V`; `targets` must have shape `[R]`
and share its device. Every target must equal `ignore_index` or lie in `[0,V)`.
`weight`, when present, must have shape `[V]`, share the device, and not require
gradients. The builtin Triton implementation currently rejects class weights,
so weighted calls resolve to `native_torch.cross_entropy`.

**Implementations**

`builtin.cross_entropy.triton` (default for supported unweighted CUDA inputs)
or `native_torch.cross_entropy`.

**Example**

```python
loss = cross_entropy(logits, labels, ignore_index=-100, reduction="mean")
```

This operation and `head_loss` are intentionally distinct. The standalone
form materializes its caller-provided logits and saves them for its VJP.
`head_loss` creates vocabulary logits in bounded chunks, immediately consumes
each chunk's loss gradient, and never retains full `[R,V]` logits.

### `HEAD_CHUNK_SCRATCH_BYTES`

**Status:** Contributor constant

`HEAD_CHUNK_SCRATCH_BYTES = 256 << 20` is the default 256 MiB target used to
derive the temporary BF16 logits chunk. It is not a persistent allocation or
model buffer.

### `default_head_chunk_size`

**Status:** Contributor helper

**Signature**

```python
default_head_chunk_size(vocab_size: int) -> int
```

**Purpose and returns**

Returns a 256-row-aligned token-chunk size targeting roughly
`HEAD_CHUNK_SCRATCH_BYTES` for BF16 logits, with a minimum of 512 rows.

**Constraints and exceptions**

`vocab_size` must be positive.

**Example**

```python
rows = default_head_chunk_size(vocab_size)
```

### `head_loss`

**Status:** Public

**Signature**

```python
head_loss(
    hidden: Tensor,
    head_weight: Tensor,
    targets: Tensor,
    *,
    chunk_size: int | None = None,
    valid_rows: int | None = None,
) -> Tensor
```

**Purpose**

Computes vocabulary projection and mean next-token cross entropy from already
normalized hidden states with a bounded chunk-local logits footprint.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `hidden` | `Tensor[...,D]`, flattened to `[R,D]` | caller-normalized hidden states |
| `head_weight` | `Tensor[V,D]` | vocabulary projection |
| `targets` | integer `Tensor[hidden.shape[:-1]]` | next-token IDs |
| `chunk_size` | positive `Optional[int]` | explicit token rows per logits chunk |
| `valid_rows` | `Optional[int]` | denominator in `[1,R]`; `None` uses `R` |

**Returns**

Scalar FP32 mean cross-entropy.

**Autograd and effects**

The chunked forward never retains `[R,V]` logits. It computes and privately
saves unit-cotangent `grad_hidden[R,D]` and `grad_head_weight[V,D]`.
Registered backward scales those tensors out-of-place by the incoming loss
cotangent and does not rerun projection or cross entropy. This supports
repeated first-order VJPs but not higher-order differentiation.

**Constraints and exceptions**

Hidden/target row counts must match; targets must lie in `[0,V)`;
`chunk_size > 0`; and `1 <= valid_rows <= R`. Detected chunked-path
violations raise `ValueError`.

**Implementations**

`builtin.head_loss.chunked` (default) or `native_torch.head_loss`, which
materializes the complete logits tensor.

**Example**

```python
normalized = rms_norm(hidden, final_norm.weight, final_norm.eps)
loss = head_loss(normalized, lm_head.weight, targets)
```

Final normalization is deliberately a normal composition. RMSNorm clients call
`rms_norm` before `head_loss`; LayerNorm clients call `layer_norm`. This keeps
the head contract independent of architecture while preserving bounded logits.
Compared with the former combined contract, ordinary autograd retains the norm
input/statistics and executes its VJP separately. Save-versus-recompute policy
belongs to graph lowering rather than to separate RMS/Layer head APIs.

## Implementation-control API

Contributor control lives in `mlops.dispatch`. Overrides are exact,
context-local, nestable, and frozen before compiler capture.

### `Implementation`

```python
Implementation(
    operation: str,
    implementation_id: str,
    provider: str,
    priority: int,
    deterministic: bool,
    supports: Callable[..., SupportResult],
    apply: Callable,
    forward: Callable | None = None,
    backward: Callable | None = None,
    estimate: Callable | None = None,
)
```

An immutable stateless record. `provider` names only the source package.
Selection and artifact manifests always use `implementation_id`. Explicit
forward/backward are either both present or both absent.

### `SupportResult`

`SupportResult.yes()` and `SupportResult.no(reason)` report support without
executing a kernel. Forced unsupported implementations fail with this reason.

### `implementation_registry`

`implementation_registry() -> Mapping[str, Mapping[str, Implementation]]`
returns a read-only snapshot. Registry records contain no tensors.

### `resolve_implementation`

```python
resolve_implementation(
    operation: str,
    *args,
    surface: str = "semantic",
    **kwargs,
) -> Implementation
```

Filters support, then selects highest priority with exact ID as a deterministic
tie-breaker. During capture it requires a frozen exact override.

### `explain_implementation`

Returns the selected/forced ID and every candidate's `SupportResult` without
running the operation.

### `use_implementation`

`use_implementation(operation, implementation_id)` temporarily forces one
exact mapping and restores an outer mapping on exit.

### `use_implementations`

`use_implementations(mapping)` temporarily forces several exact operation
mappings. Unknown operation/ID pairs fail before entering the context.

### `capture_dispatch`

Captures exact implementation call counts during one isolated warmup. The
resolver records automatically; semantic functions contain no trace calls.

### `dispatch_manifest`

Converts a trace into one exact operation-to-implementation mapping and rejects
an operation that exercised multiple implementations.

### `implementation_pairs`

Builds exact default-versus-native-Torch mappings for verifier sweeps.

### `CostHints`

```python
CostHints(
    logical_flops: int | None = None,
    logical_bytes_accessed: int | None = None,
    implementation_flops: int | None = None,
    implementation_bytes_accessed: int | None = None,
    workspace_bytes: int | None = None,
    notes: tuple[str, ...] = (),
)
```

`None` means undefined; zero means known-zero. Workspace is scratch whose
lifetime ends when this single forward/backward call returns. It excludes
caller/module inputs, returned outputs, autograd residuals, reusable tables,
allocator cache, and runtime leeway.

### `estimate_implementation`

Returns canonical logical hints merged with the exact implementation's optional
physical/workspace hints. Estimation examines metadata only and retains no
tensors. Profiling remains authoritative.

### `GradcheckCase`

Defines one semantic operation invocation and optional exact implementation
IDs for batch gradcheck.

### `GradcheckResult`

Reports `passed`, `failed`, or `skipped` plus a reason.

### `gradcheck_implementation`

Calls `torch.autograd.gradcheck` for one exact implementation. Unsupported
float64/device/layout inputs are skipped; another implementation is never used.

### `gradcheck_implementations`

Runs a sequence of `GradcheckCase` values in deterministic implementation-ID
order.

### Exact implementation registry

| Operation | Default | Other registered implementations |
|---|---|---|
| `cross_entropy` | `builtin.cross_entropy.triton` | `native_torch.cross_entropy` |
| `embedding` | `builtin.embedding.deterministic` | `native_torch.embedding` |
| `gelu` | `builtin.gelu.aten_explicit_backward` | `native_torch.gelu` |
| `rms_norm` | `builtin.rms_norm.triton` | `liger.rms_norm`, `native_torch.rms_norm` |
| `layer_norm` | `builtin.layer_norm.triton` | `native_torch.layer_norm` |
| `swiglu` | `builtin.swiglu.triton` | `builtin.swiglu.triton_legacy`, `native_torch.swiglu` |
| `packed_swiglu` | `builtin.packed_swiglu.triton` | `builtin.packed_swiglu.triton_legacy`, `native_torch.packed_swiglu` |
| `rope` | `builtin.rope.triton_table` | `builtin.rope.triton_analytic`, `native_torch.rope` |
| `partial_rope` | `builtin.partial_rope.triton` | `native_torch.partial_rope` |
| `flash_attention` | `builtin.flash_attention.aten` | `native_torch.flash_attention` |
| `mla_attention` | `builtin.mla_attention.flash` | `native_torch.mla_attention` |
| `head_loss` | `builtin.head_loss.chunked` | `native_torch.head_loss` |
| `causal_conv_silu` | `fla.causal_conv_silu` | `native_torch.causal_conv_silu` |
| `l2_norm` | `fla.l2_norm` | `native_torch.l2_norm` |
| `gated_rms_norm` | `fla.gated_rms_norm` | `native_torch.gated_rms_norm` |
| `linear_attention` | `fla.linear_attention.gated_delta_rule` | `native_torch.linear_attention` |
| `linear_mixer` | `builtin.linear_mixer.packed` | `native_torch.linear_mixer.per_sequence` |
| `dsa_attention` | `builtin.dsa_attention.sparse` | `native_torch.dsa_attention` |
| `dsa_indexer_key_norm` | `builtin.dsa_indexer_key_norm.layer_norm` | `native_torch.dsa_indexer_key_norm` |
| `index_scores` | `builtin.index_scores.einsum` | `native_torch.index_scores` |
| `moe` | `builtin.moe.grouped_gemm_composed` | `builtin.moe.grouped_gemm`, `scattermoe.moe`, `native_torch.moe` |

## Private dispatcher boundary

Opaque implementations register stable functional custom-op targets in the
`mlops` namespace. These targets are private to the
implementation adapter: model and block code calls the semantic façade, while
manual runtimes call the explicit façade.

Each owning provider module keeps its forward custom op, fake implementation,
`register_autograd` setup, and optional backward custom op together. Residual
outputs are visible to FakeTensor and AOTAutograd but hidden from model code.
Their exact schema is implementation-specific and is therefore documented in
the provider source rather than frozen as another public ABI.

All dispatcher schemas are functional (`mutates_args=()`), and the contract
suite runs `torch.library.opcheck` over every registered opaque forward. Target
names include operation, provider, and variant—for example
`rms_norm_liger_fwd` and `moe_scattermoe_fwd`—so captured graphs retain exact
implementation identity.

## Operation-package example

The following shows the intended dependency direction:

```python
import torch
import torch.nn as nn

from mlops import rms_norm, swiglu


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden_width: int):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(width))
        self.gate = nn.Linear(width, hidden_width, bias=False)
        self.up = nn.Linear(width, hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = rms_norm(hidden, self.norm_weight)
        return hidden + self.down(swiglu(self.gate(normalized), self.up(normalized)))
```

The module owns parameters and composition. The operation package owns tensor
semantics, implementation selection, registered dispatcher boundaries, and
VJPs.

## Explicit operations

This section specifies the public, autograd-independent operation surface in
`mlops.explicit`. It complements the
model-facing API above: both surfaces use the same implementations, but this
one exposes forward residuals and VJP arguments directly for callers that
manage execution and tensor lifetimes.

### Contract

Every module exports exactly two functions:

```python
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

The current surface allocates and returns fresh output tensors. A future
standalone operations package may add a separate caller-owned-buffer launch
ABI without changing these mathematical signatures. Such an execution-only
`out=` surface must not replace the differentiable functional custom ops used
by models.

These entry points use the same implementation resolver and raw implementation math
as the semantic surface; they are not a second copy of dispatch. Select an
exact implementation with `use_implementation` or a frozen manifest around
both calls. Apply-only native-Torch graphs are rejected by the explicit
surface because they rely on ordinary autograd rather than a raw VJP.

### Import and calling convention

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

The semantic API remains unchanged:

```python
from mlops import rms_norm

output = rms_norm(x, weight)  # ordinary registered autograd
```

Models and blocks use the semantic API. Manual runtimes, isolated kernel
tests, and profilers may use the explicit API. Blocks and models remain
forward-only and intentionally expose no hand-authored backward methods.

### Tensor conventions

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

### Quick index

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
| `causal_conv_silu` | convolution output | caller retains cumulative sequence lengths | input, weight |
| `linear_attention` | recurrent output | FLA gate/matrix and sequence metadata | seven continuous inputs |
| `moe` | routed output and diagnostics | complete route/dispatch state | hidden, residual, router, expert weights |
| `head_loss` | scalar loss | seed-one hidden/head VJPs | scaled hidden/head VJPs |

### Embedding

#### `explicit.embedding`

**Signatures**

```python
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
`num_embeddings`.

### Normalization

#### `explicit.rms_norm`

**Signatures**

```python
forward(
    x: Tensor[...,D],
    weight: Tensor[D],
    eps: float = 1e-5,
) -> tuple[Tensor[...,D], Tensor[R]]

backward(
    grad_output: Tensor[...,D],
    x: Tensor[...,D],
    weight: Tensor[D],
    rstd: Tensor[R],
) -> tuple[Tensor[...,D], Tensor[D]]
```

Forward returns `(output, rstd)`. `rstd` is FP32 and is the only
operation-produced residual. Backward returns `(grad_x, grad_weight)`.

#### `explicit.layer_norm`

**Signatures**

```python
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

#### `explicit.l2_norm`

**Signatures**

```python
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

#### `explicit.gated_rms_norm`

**Signatures**

```python
forward(
    x: Tensor[...,D],
    gate: Tensor[...,D],
    weight: Tensor[D],
    eps: float = 1e-5,
) -> tuple[Tensor[...,D], Tensor[R]]

backward(
    grad_output: Tensor[...,D],
    x: Tensor[...,D],
    gate: Tensor[...,D],
    weight: Tensor[D],
    rstd: Tensor[R],
    eps: float = 1e-5,
) -> tuple[Tensor[...,D], Tensor[...,D], Tensor[D]]
```

The operation computes `silu(gate) * RMSNorm(x) * weight`. Backward returns
`(grad_x, grad_gate, grad_weight)`.

### Activations and rotary position

#### `explicit.gelu`

```python
forward(x: Tensor[*]) -> Tensor[*]
backward(grad_output: Tensor[*], x: Tensor[*]) -> Tensor[*]
```

Applies tanh-approximate GELU and its explicit ATen VJP. Retain `x` for
backward.

#### `explicit.swiglu`

```python
forward(
    gate: Tensor[...,F],
    up: Tensor[...,F],
) -> Tensor[...,F]

backward(
    grad_output: Tensor[...,F],
    gate: Tensor[...,F],
    up: Tensor[...,F],
) -> tuple[Tensor[...,F], Tensor[...,F]]
```

Returns `silu(gate) * up`; backward returns `(grad_gate, grad_up)`. Force
`builtin.swiglu.triton` or the historical
`builtin.swiglu.triton_legacy` around both calls when selecting a variant.

#### `explicit.packed_swiglu`

```python
forward(packed: Tensor[R,2F]) -> Tensor[R,F]
backward(
    grad_output: Tensor[R,F],
    packed: Tensor[R,2F],
) -> Tensor[R,2F]
```

The first half of `packed` is the gate and the second is the up projection.

#### `explicit.rope`

```python
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

#### `explicit.partial_rope`

```python
forward(x, positions, base, rotary_dim, cosine, sine) -> Tensor[T,H,Dh]
backward(grad_output, positions, base, rotary_dim, cosine, sine) -> Tensor[T,H,Dh]
```

Only the leading `rotary_dim` channels are rotated. Remaining channels are
copied in forward and backward.

### Attention and sequence operations

#### `explicit.flash_attention`

```python
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
`cu_seqlens` contains caller-constructed packed boundaries. Unlike the semantic
surface, the explicit interface does not prepare this metadata for the caller.
The explicit surface currently accepts `builtin.flash_attention.aten`; apply-only
`native_torch.flash_attention` is rejected rather than replaying autograd.

#### `explicit.dsa_attention`

```python
forward(q, k, v, indices, lengths) -> tuple[output, lse]
backward(grad_output, q, k, v, indices, lse, lengths)
    -> tuple[grad_q, grad_k, grad_v]
```

`q` and `k` have shape `[T,Hq,Dqk]`, `v` has `[T,Hq,Dv]`, and `indices` has
`[T,K]`. Selected indices are discrete fixed inputs and receive no gradient.

#### `explicit.causal_conv_silu`

```python
forward(
    x: Tensor[T,C],
    weight: Tensor[C,1,Wc],
    lengths: Sequence[int],
    cumulative_lengths: Tensor[*],
) -> Tensor[T,C]

backward(
    grad_output: Tensor[T,C],
    x: Tensor[T,C],
    weight: Tensor[C,1,Wc],
    cumulative_lengths: Tensor[*],
) -> tuple[Tensor[T,C], Tensor[C,1,Wc]]
```

Convolution state resets at every packed boundary. Caller-owned cumulative
metadata may be an empty tensor for a single sequence and must be passed to
forward and backward unchanged.

#### `explicit.linear_attention`

```python
forward(
    q, k, v, beta, a, a_log, dt_bias, lengths,
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

### Mixture of experts

#### `explicit.moe`

**Forward signature**

```python
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

```python
backward(
    grad_output, grad_aux, grad_probability_sum,
    h2, router_weight, w13_experts, w2_experts,
    logits, route_weights, route_ids, order, offsets, slots, h13,
    *, top_k: int, routing_mode: str, lengths: Sequence[int],
    swiglu_implementation: str = "builtin.packed_swiglu.triton",
) -> tuple[grad_h2, grad_residual, grad_router, grad_w13, grad_w2]
```

`h2` and `residual` have `[...,D]`; `router_weight` is `[D,E]`;
`w13_experts` is `[E,D,2F]`; and `w2_experts` is `[E,F,D]`.
Backward requires cotangents for the differentiable public `output`, `aux`, and
`probability_sum` roots. `counts` and discrete routing/dispatch metadata have
no VJP. Forward and backward must use the same `top_k`, `routing_mode`, packed
lengths, and builtin routed-SwiGLU implementation.

This atomic operation covers the routed expert path only. Shared-expert
composition remains ordinary block/model computation and is not reproduced in
this explicit entry point.

### Losses

#### `explicit.cross_entropy`

**Signatures**

```python
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

### Chunked language-model losses

The head operation uses a memory-bounded one-pass contract: forward creates
each chunk-local logits tensor once, immediately forms its cross-entropy
gradient, and returns seed-one parameter/input VJPs. Its backward entry point
only scales those immutable seeds by the caller-supplied scalar loss
cotangent.

#### `explicit.head_loss`

```python
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

### Operations without explicit VJPs

Not every model-facing helper is an atomic differentiable operation:

| Semantic helper | Reason there is no explicit module |
|---|---|
| `mla_attention` | composition of padding/slicing and explicit flash attention |
| `linear_mixer` | graph-layout/control-flow selection around a caller function |
| `dsa_indexer_key_norm`, `index_scores` | ordinary differentiable PyTorch compositions |
| `topk_indices`, `selection_mask` | discrete selection with no mathematical VJP |
| implementation/diagnostic context managers | execution policy rather than tensor operations |

The package does not hand-author composite block or model backward functions.
Those remain forward-only PyTorch compositions; an external AOT lowering layer
may generate stage graph pairs when needed.

### Adding an explicit operation

When adding an opaque differentiable `*_fwd` custom op:

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

The coverage test intentionally fails if an opaque forward custom op exists
without a corresponding explicit module.
