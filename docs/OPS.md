# Operations API

Semantic tensor calls with ordinary PyTorch autograd. Each entry specifies
shapes, dtypes, effects, implementations and examples. Separate references cover
[explicit forward/VJP calls](EXPLICIT_OPS.md) and [implementation selection](DISPATCH.md).

## Contents

- [API scope](#api-scope)
- [Reference-entry format](#reference-entry-format)
- [Tensor conventions](#tensor-conventions)
- [Quick index](#quick-index)
- [Embedding](#embedding)
- [Normalization and activations](#normalization-and-activations)
- [Rotary position operations](#rotary-position-operations)
- [Dense and latent attention](#dense-and-latent-attention)
- [DSA indexing and selected attention](#dsa-indexing-and-selected-attention)
- [Hybrid linear-attention operations](#hybrid-linear-attention-operations)
- [Mixture of experts](#mixture-of-experts)
- [Hyper-connections](#hyper-connections)
- [Language-model epilogues](#language-model-epilogues)
- [Implementation-control API](DISPATCH.md#implementation-control-api)
- [Private dispatcher boundary](PROVIDERS.md#private-dispatcher-boundary)
- [Operation-package example](PROVIDERS.md#operation-package-example)
- [Explicit operations](EXPLICIT_OPS.md#explicit-operation-reference)

## API scope

The stable model-facing import surface is:

```python
from mlops import (
    build_rope_tables,
    causal_conv_silu,
    causal_conv_silu_qkv,
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
    kda_decay,
    kimi_delta_attention,
    l2_norm,
    layer_norm,
    linear_attention,
    linear_mixer,
    lora_head_loss,
    mhc_coefficients,
    mhc_combine,
    mhc_normalize_streams,
    mhc_pre,
    moe,
    packed_swiglu,
    partial_rope,
    pooled_topk,
    prepare_packed_sequence_metadata,
    rms_norm,
    rope,
    selection_mask,
    sigmoid_topk,
    sparse_latent_attention,
    sparse_mla,
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

Full reference entries use these fields:

1. **Signature** — callable surface.
2. **Purpose** — semantic behavior.
3. **Parameters** — types, shapes, and meanings.
4. **Returns** — public return values and shapes.
5. **Autograd and effects** — gradients, saved state, mutation, and aliasing.
6. **Constraints and exceptions** — required relationships and expected errors.
7. **Implementations** — exact registered implementation IDs.
8. **Example** — minimal canonical use.

Compact entries give the signature, shapes, gradients, and implementations in
prose.

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
tensor input. Dense attention instead takes cumulative offsets `cu_seqlens`
as a device tensor, and causal convolution and gated linear attention take
caller-owned device metadata prepared once per packed round by
`prepare_packed_sequence_metadata`; providers never derive host counts from
CUDA tensors or retain metadata caches.

### Mutation, aliasing, and higher-order gradients

All registered model-facing differentiable operations declare
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
| `rope`, `partial_rope` | position | shape of input | `builtin.rope.triton_table` / `builtin.partial_rope.triton` |
| `build_rope_tables` | reusable position tables | FP32 cosine/sine `[rows,rotary_dim]` | construction helper |
| `flash_attention` | dense attention | `[T,Hq,Dh]` | `builtin.flash_attention.aten` |
| `index_scores`, `topk_indices` | DSA selection | `[T,T]`, `[T,K]` | `builtin.index_scores.einsum` / not dispatched |
| `dsa_indexer_key_norm` | indexer normalization | shape of key | `builtin.dsa_indexer_key_norm.layer_norm` |
| `selection_mask` | selected causal mask | FP32 `[query_rows,key_rows]`, 0 or -inf | construction helper |
| `dsa_selection_override` | diagnostic selection | context manager | caller-supplied selection |
| `dsa_attention` | selected attention | `[T,Hq,Dv]` | `builtin.dsa_attention.sparse` |
| `sparse_latent_attention`, `sparse_mla` | tied-latent attention / composition | `[T,H,L]` / `[T,H,Dv]` | `builtin.sparse_latent_attention.hopper` on SM90 / TileLang fallback; composed |
| `pooled_topk` | pooled causal selection | `[T,K+pool_size-1]` | `cudnn.pooled_topk.sm90` / Triton fallback |
| `kimi_delta_attention`, `kda_decay` | recurrent attention / FP32 decay | `[T,H,D]` | cuLA SM90 / FLA fallback; composed decay |
| `sigmoid_topk` | expert selection | IDs and weights `[T,K]` | composed PyTorch |
| `mhc_pre` | residual streams | post coefficients / mixing / branch input | DeepGEMM fused pre / TileKernels Sinkhorn on SM90/SM100; native fallback |
| `mhc_combine` | residual streams | mixed streams | TileKernels post on SM90/SM100; builtin/native fallback |
| `mhc_normalize_streams`, `mhc_coefficients` | composition helpers | normalized streams / coefficients | composed PyTorch |
| `causal_conv_silu` | hybrid mixer | shape of input | `fla.causal_conv_silu` |
| `causal_conv_silu_qkv` | hybrid mixer | three projection shapes | `fla.causal_conv_silu_qkv` for supported frozen weights |
| `linear_attention` | hybrid mixer | shape of V | `fla.linear_attention.gated_delta_rule` |
| `linear_mixer` | graph-layout policy | shape of hidden | `builtin.linear_mixer.packed` |
| `prepare_packed_sequence_metadata` | reusable sequence metadata | cumulative lengths and chunk indices | construction helper |
| `moe` | routed experts | output + diagnostics | `sonicmoe.moe.composed` on 9.x/10.x / `builtin.moe.grouped_gemm_composed` |
| `cross_entropy` | training loss | FP32 `[R]` or scalar | `builtin.cross_entropy.triton` |
| `head_loss` | bounded-logits training epilogue | FP32 scalar | `builtin.head_loss.chunked` |
| `lora_head_loss` | bounded-logits low-rank head | FP32 scalar | `builtin.lora_head_loss.chunked` |

## Embedding

### `embedding`

**Status:** Public

**Signature**

```text
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
`grad_weight[V,D]`, at the builtin implementation's
[`weight_gradient_dtype`](DISPATCH.md#selection-and-precision) when
one is set; `tokens` is saved as non-differentiable metadata. No input is
mutated or aliased.

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

```text
rms_norm(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-5,
    *,
    zero_centered: bool = False,
) -> Tensor
```

**Purpose**

Applies RMS normalization over the final dimension followed by an affine gain.
By default the normalized value is rounded to `x`'s dtype before the gain, as
Llama's norm does. A zero-centered weight, as Qwen 3.5 keeps every norm but its
linear attention's, is an offset from one: the FP32 normalized value is scaled
by `1 + weight` before the one cast, so stored weights stay small and the scale
keeps full precision. The gradients follow the same arithmetic.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | `Tensor[...,D]` | rows to normalize |
| `weight` | `Tensor[D]` | learned gain, or its offset from one when zero-centered |
| `eps` | `float` | positive squared-RMS stabilizer |
| `zero_centered` | `bool` | scale by `1 + weight` in FP32 before the cast |

**Returns**

`Tensor[...,D]` with the shape and dtype of `x`.

**Autograd and effects**

The optimized forward additionally exposes a private FP32 `rstd[R]` residual
and saves `x`, `weight`, and `rstd`. Backward returns gradients for
`x` and `weight`, the builtin implementation's `weight` gradient at
[`weight_gradient_dtype`](DISPATCH.md#selection-and-precision) when
one is set. No input is mutated.

**Constraints and exceptions**

`weight.numel()` must equal `x.shape[-1]`; `x` must have a nonempty final
dimension. Backend shape errors are raised for incompatible inputs.

**Implementations**

`builtin.rms_norm.triton` (default) or `native_torch.rms_norm`.

**Example**

```python
normalized = rms_norm(hidden, norm.weight, eps=1e-5)
```

### `layer_norm`

**Status:** Public

**Signature**

```text
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
present, the builtin implementation's `grad_weight` and `grad_bias` at
[`weight_gradient_dtype`](DISPATCH.md#selection-and-precision) when
one is set. No input is mutated.

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

```text
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

For strided inputs with feature width up to 512, `fla.l2_norm` reads the input
strides directly and writes contiguous normalized output. This combines layout
conversion with normalization for Q/K views of packed QKV tensors.

### `gated_rms_norm`

**Status:** Public

**Signature**

```text
gated_rms_norm(
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    eps: float = 1e-5,
    *, activation: str = "silu",
) -> Tensor
```

**Purpose**

Computes gated RMS normalization using SiLU (default) or sigmoid gating, with FP32 normalization/gate arithmetic and output in the input dtype.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `x` | `Tensor[...,D]` | values to normalize |
| `gate` | `Tensor[...,D]` | elementwise gate input |
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


`activation` accepts `"silu"` or `"sigmoid"`; other values raise `ValueError`. Use the same choice in explicit forward and backward.

### `swiglu`

**Status:** Public

**Signature**

```text
swiglu(gate: Tensor, up: Tensor, *, limit: float | None = None) -> Tensor
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
Triton VJP rounds as PyTorch's CUDA SiLU backward does: an accurate exponential
and round-to-nearest division for the sigmoid `s`, and the left-associated
`(g * s) * (1 + x * (1 - s))` for the gate cotangent `g`. `triton_legacy` uses
Triton's `tl.sigmoid` and the right-associated `g * (s * (1 + x * (1 - s)))`,
for ablation only.

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


`limit=None` preserves unclipped behavior. A finite positive `limit` clips gate values above the limit and up values to `[-limit, limit]`; derivatives include the clamp masks. The Triton path fuses this into its forward/backward kernels. Use the same limit in both explicit calls.

### `packed_swiglu`

**Status:** Public

**Signature**

```text
packed_swiglu(packed: Tensor, *, limit: float | None = None) -> Tensor
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

Saves `packed`; backward returns one `grad_packed[...,2F]`. Its
implementations mirror `swiglu`'s and are selected separately.

**Constraints and exceptions**

The final width must be even.

**Implementations**

`builtin.packed_swiglu.triton` (default),
`builtin.packed_swiglu.triton_legacy`, or `native_torch.packed_swiglu`.

**Example**

```python
activated = packed_swiglu(project(hidden))  # project width is 2F
```


`limit=None` preserves unclipped behavior. A finite positive `limit` clips gate values above the limit and up values to `[-limit, limit]`; derivatives include the clamp masks. The Triton path fuses this into its forward/backward kernels. Use the same limit in both explicit calls.

### `gelu`

**Status:** Public

**Signature**

```text
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

```text
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

```text
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

```text
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

Saves `positions`, `cosine`, `sine`, and scalar metadata; no activation.
Backward inverse-rotates the leading gradient channels and passes the tail
through.

**Constraints and exceptions**

`0 < rotary_dim <= Dh`, and `rotary_dim` must be even. Token metadata must
match the flattened token extent.

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

```text
flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    deterministic: bool | None = None,
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
| `cu_seqlens` | `Tensor[N+1]` | int32 cumulative sequence offsets on the queries' device, from `0` to `T`; a repeated offset is an empty sequence |
| `max_seqlen` | `int` | an upper bound on any one sequence's length |
| `causal` | `bool` | whether future keys are masked |
| `softmax_scale` | `Optional[float]` | explicit logit scale; `None` uses the backend default |
| `deterministic` | `Optional[bool]` | whether the backward accumulates in a fixed order; `None` follows `deterministic_kernels` |

**Returns**

`Tensor[T,Hq,Dh]`.

**Autograd and effects**

The optimized registered path saves Q/K/V, output, `cu_seqlens`, and FP32
log-sum-exp when the native flash primitive supplies it; it retains no tensor
state. Backward returns gradients for Q/K/V.

The offsets are data, not structure: a captured graph takes `cu_seqlens` as
an input, so one graph serves every packing of the same `T` tokens. To keep
its shape fixed across packings, pad it by repeating the last offset; each
repeat is an empty sequence. Inputs are made
contiguous when needed; caller tensors are not modified.

Under FA3 the variable-length backward accumulates dQ across key blocks with
atomics, so its sum order follows block completion and one step run twice from
one seed can end on different gradients. `deterministic=True` accumulates in a
fixed order instead, at a throughput cost, which is why the default is off.
`None`, the default, follows whatever `mlops.dispatch.deterministic_kernels`
is in effect. The request is resolved at call time and baked into any captured
graph, so a graph traced under it stays deterministic on replay. It is baked
as a constant with no guard, which is what keeps the read from breaking a
`fullgraph` capture, so the converse also holds: a compiled graph cached from
a run without the request is reused unchanged rather than recompiled under it.
A caller that compiles across the boundary wants a compile cache per setting. Requesting it
on a Hopper GPU without the `flash-attn-3` wheel raises `RuntimeError` rather
than quietly returning unordered gradients. FA3's kernels do not run on other
GPUs, where the request falls back to the aten backward without the ordering
guarantee. `native_torch.flash_attention` is ordered already and ignores the
request.

**Constraints and exceptions**

- Q/K/V must be rank three.
- `k.shape == v.shape`.
- Q and K head widths must match.
- `Hq % Hkv == 0`.
- `cu_seqlens` must be one-dimensional int32 on the queries' device, as the
  kernel reads it; nothing converts it. It must start at `0`, never decrease,
  and end at `T`, with no sequence longer than `max_seqlen`. Its values stay
  on the device and are not read on the host, so these are the caller's to
  guarantee.

Offsets that are not one-dimensional int32 on the queries' device leave no
implementation supported, so dispatch raises `RuntimeError`. The shape
relationships are left to the kernels, which raise their own errors.

**Implementations**

`builtin.flash_attention.aten` (default) or
`native_torch.flash_attention`. The builtin implementation requires PyTorch's
native variable-length FlashAttention primitive and rejects unsupported calls;
it never silently falls back.

On a Hopper GPU with the `flash-attn-3` wheel installed, the builtin
implementation activates PyTorch's FlashAttention-3 registration on first
use, so the same aten primitive runs FA3 kernels. The registry identity
remains `builtin.flash_attention.aten`; on other devices, or without the
wheel, the primitive keeps its stock kernels.

**Example**

```python
cu_seqlens = torch.tensor([0, 731, 943, 1024, 1024], dtype=torch.int32, device=q.device)
attended = flash_attention(q, k, v, cu_seqlens, 2048)  # [T,Hq,Dh], T=1024
```

### `sparse_mla`

`sparse_mla(query, latent, key_projection, value_projection, indices, *, scale)`
absorbs the key projection into queries, calls tied-latent sparse attention,
then applies the value projection. `query[T,H,Dqk]`, `latent[T,L]`,
`key_projection[H,Dqk,L]`, and `value_projection[H,Dv,L]` are caller-owned.
Returns `[T,H,Dv]`; gradients cover queries, latent and both projection tensors.
Use the original query/key attention `scale`, not the latent width. Projection
absorption can change BF16 intermediate rounding relative to expanded K/V.

## DSA indexing and selected attention

### `dsa_indexer_key_norm`

**Status:** Public

**Signature**

```text
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

```text
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

```text
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

```text
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

```text
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

```text
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

### `sparse_latent_attention`

`sparse_latent_attention(q, kv, indices, *, scale)` accepts BF16
`q[T,H,L]` and one shared `kv[T,L]` bank. It returns `[T,H,L]`.
`indices[T,K]` contains unique causal key indices per sequence, with -1 padding;
each query needs at least one key. H must be divisible by 16 and L one of
64/128/256/512. `scale` is explicit. Backward returns dQ and the combined dKV.
On SM90, for `L=512` and `H` a multiple of 64 with TileLang and cuDNN
Frontend's DSA module installed, `builtin.sparse_latent_attention.hopper` is
preferred: FlashMLA's sparse prefill forward when FlashMLA is installed, else
TileLang's pipelined forward, and cuDNN Frontend's DSA backward. Every other
call uses `builtin.sparse_latent_attention.tilelang`, whose backward
accumulates dKV with FP32 atomics; neither guarantees bitwise deterministic
gradients.

Optional `q_rope[T,H,64]` and `k_rope[T,64]` must be supplied together, in the
same BF16 dtype/device. Their already-rotated dot product contributes to the
score, while values/output remain latent-width. This selects
`builtin.sparse_latent_attention.rope.hopper` under the same conditions, else
`builtin.sparse_latent_attention.rope.tilelang`; autograd also produces both
positional gradients. The caller applies RoPE and chooses the combined
query/key scale.

### `pooled_topk`

`pooled_topk(q, key, gates, head_weights, position_bias, boundaries, *,
top_k=2048)` selects pooled causal keys.
CUDA inputs are `q[T,H,D]`, `key[T,D]`, `gates[T,D]`,
`head_weights[T,H]`, and `position_bias[pool_size,D]`.
`boundaries` is an integer CPU tensor with sequence boundaries, passed as runtime
data. Output is INT32 `[T,top_k+pool_size-1]`, padded with -1.
Selection is non-differentiable; short sequences select all causal keys without
scoring. On SM90, `cudnn.pooled_topk.sm90` is preferred for BF16 Q/K, D=128
and H=16/32/64 with supported pool/top-k geometry. It preserves FP32 head
weights and caps the combined scoring/radix scratch at 256 MiB; output indices
are additional. Other geometries use `builtin.pooled_topk.triton`, which pools the keys of each
sequence once, then one Triton launch per chunk of queries scores the chunk
against every pooled key with the scale, ReLU and head weights applied in
registers, and a second launch reads each query's FP32 scores and writes its
selection, the cutoff found by a binary search over the float bit patterns and
ties going to the lowest pool. A chunk's scores stay within 256 MiB, and a query
wider than 16384 pools selects with `torch.topk` instead. Queries and keys that
share a 16-bit dtype are scored on the tensor cores; any other dtypes are scored
in FP32, which they reach exactly; the width may be at most 256. Ties need not
match another top-k implementation, and the heads sum in another order than in
PyTorch arithmetic, so a cutoff between scores closer than FP32 rounding may
select either.

## Hybrid linear-attention operations

### `causal_conv_silu`

**Status:** Public

**Signature**

```text
causal_conv_silu(
    x: Tensor,
    weight: Tensor,
    cumulative: Tensor,
    chunk_indices: Tensor | None = None,
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
| `cumulative` | INT64 `Tensor[L+1]`, or empty for one sequence | caller-owned cumulative boundaries |
| `chunk_indices` | optional INT64 `Tensor[Nchunk,2]`, or empty for one sequence | caller-owned `(sequence, chunk)` map; supplying it avoids reconstructing it inside provider kernels |

**Returns**

A tensor matching `x`.

**Autograd and effects**

The FLA path saves `x`, `weight`, and the caller-owned cumulative sequence
metadata. Backward returns gradients for `x` and `weight`. The operation does
not construct or retain metadata. The metadata is data: a captured graph takes
it as an input, so one graph serves every packing of the same tokens.

**Constraints and exceptions**

The metadata must describe the flattened row count; its values stay on the
device, so that is the caller's to guarantee. Weight channels must match `C`.

**Implementations**

`fla.causal_conv_silu` (default) or `native_torch.causal_conv_silu`.

**Example**

```python
cumulative, chunks = prepare_packed_sequence_metadata(lengths, projected)
convolved = causal_conv_silu(projected, conv.weight, cumulative, chunks)
```

The autograd path avoids weight-gradient computation when `weight.requires_grad`
is false. Its input-only backward handles widths up to eight without a global
preactivation buffer; wider widths fall back to FLA. The explicit backward
returns both input and weight gradients.

### `causal_conv_silu_qkv`

**Status:** Public semantic operation; ordinary autograd, no explicit VJP façade.

**Signature**

```text
causal_conv_silu_qkv(q, k, v, weight, cumulative, chunk_indices=None)
    -> tuple[Tensor, Tensor, Tensor]
```

**Purpose**

The concatenate/convolve/split operation on three projections, with sequence
resets and SiLU identical to `causal_conv_silu`. This boundary lets backward
consume separate output gradients and write projection gradients directly.
Projection weights, Q/K normalization and attention gates stay outside it.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `q`, `k`, `v` | `[T,Cq]`, `[T,Ck]`, `[T,Cv]` | same token count, dtype and device; channel widths may differ |
| `weight` | `[Cq+Ck+Cv,1,Wc]` | Q, then K, then V convolution channels |
| `cumulative`, `chunk_indices` | as in `causal_conv_silu` | caller-owned sequence boundaries and optional chunk map |

**Returns**

Three tensors matching the projection shapes. They may be disjoint views of
one newly allocated convolution output; none aliases a caller input.

**Autograd and effects**

For frozen convolution weights, the FLA implementation saves the packed
projection input, weight and sequence metadata. Its private input VJP reads
separate Q/K/V gradients and returns three contiguous gradients, avoiding a
packed gradient allocation and subsequent projection-layout copies. FP32
accumulation and the FLA preactivation rounding boundary are preserved.
Trainable convolution weights use the ordinary differentiable composition.
No caller tensor is mutated. Higher-order gradients are not supported.

**Constraints and exceptions**

Inputs must be matrices with positive channel widths. Invalid shape/device
relationships raise `ValueError`. Metadata must describe the token rows, as
for `causal_conv_silu`; metadata contents are not copied to the host to validate.
The optimized input VJP supports CUDA BF16/FP16/FP32 and frozen kernels of
width at most eight. Other cases retain the ordinary convolution composition.

**Implementations**

`fla.causal_conv_silu_qkv` (priority 100, supported frozen weights) and
`builtin.causal_conv_silu_qkv.composed` (portable fallback, priority 0).
The FLA provider wraps three existing convolution calls in an opaque multi-output
forward with registered autograd, so `torch.export` and later AOT differentiation
preserve the separate-gradient VJP. A private packed-input residual is saved for
backward. Both operators register fake behavior and logical FLOP formulas. The
portable fallback is the ordinary differentiable concatenate/convolve/split
composition. Canonical cost estimates count the same arithmetic as packed
convolution and exclude layout copies; `weight_gradient=False` excludes the
frozen-weight gradient work.

**Example**

```python
q, k, v = causal_conv_silu_qkv(
    q_proj(x), k_proj(x), v_proj(x), conv.weight, cumulative, chunks
)
q, k = l2_norm(q.reshape(T, H, D)), l2_norm(k.reshape(T, H, D))
```

### `prepare_packed_sequence_metadata`

**Status:** Public preparation helper

**Signature**

```text
prepare_packed_sequence_metadata(
    lengths: Sequence[int] | Tensor,
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

`lengths` may instead be a one-dimensional integer tensor on `like.device`,
zero-padded to a fixed number of sequences `N`. The lengths are then data: the
metadata comes from graph-visible tensor arithmetic in shapes fixed by `like`
and `N` alone, so one captured graph serves every packing of the same tokens.
`like` must then be the packed tokens, whose count `T` bounds the chunks:
`cumulative` has `N + 2` entries and `chunk_indices` has
`ceil(T / chunk_size) + N` rows, one for every chunk the sequences could need.
The extra offset closes one more, empty sequence, and the rows past the real
chunks belong to it, so kernels read and write nothing for them.

**Constraints and exceptions**

Integer lengths and `chunk_size` must be positive; a lengths tensor must be
nonnegative and sum to `T`, which, being on the device, is the caller's to
guarantee. Callers pass the same returned
tensors to every causal-convolution and linear-attention layer in the round.
Providers consume but never retain or reconstruct them.

**Example**

```python
cumulative, chunks = prepare_packed_sequence_metadata(lengths, hidden)
```

### `linear_attention`

**Status:** Public

**Signature**

```text
linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    a: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
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
| `cumulative` | INT64 `Tensor[L+1]`, or empty for one sequence | caller-owned cumulative boundaries |
| `chunk_indices` | INT64 `Tensor[Nchunk,2]`, or empty for one sequence | caller-owned `(sequence, chunk)` map |
| `scale` | `Optional[float]` | query/key scale; default `Dk**-0.5` |

**Returns**

`Tensor[T,Hv,Dv]`, matching `v`.

**Autograd and effects**

The FLA boundary saves all differentiable inputs, the caller-owned metadata,
and private gate/matrix residuals. Backward returns seven tensor gradients.
`a_log` and `dt_bias` gradients are FP32 in the fake/FLA contract. Neither the
operation nor provider constructs or retains sequence metadata. The metadata
is data: a captured graph takes it as an input, so one graph serves every
packing of the same tokens.

FLA forward releases its private WY factors after state reconstruction;
backward releases its private replay factors and recurrent histories after their
final consumers. This reduces simultaneous scratch storage without changing
the saved gate/matrix, kernels or numerical defaults. See
[kernel lifetime ownership](KERNELS.md#gated-deltanet-scratch-lifetimes).

**Constraints and exceptions**

Token axes and value-head gating dimensions must agree. The metadata must
cover `T`. Multi-sequence calls require matching INT64 cumulative and 64-token
chunk-index tensors; single-sequence calls use empty sentinels in both.
Provider-specific unsupported shapes fail during implementation resolution.

**Implementations**

`fla.linear_attention.gated_delta_rule` (default) or
`native_torch.linear_attention` recurrent oracle.

**Example**

```python
cumulative, chunks = prepare_packed_sequence_metadata(lengths, q)
values = linear_attention(
    q, k, v, beta, a, A_log, dt_bias, cumulative, chunks
)
```

### `linear_mixer`

**Status:** Public policy helper

**Signature**

```text
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

### `kimi_delta_attention`

`kimi_delta_attention(q, k, v, g, beta, cumulative, chunks)` returns packed
`[T,H,Dv]` KDA output. Inputs `q`, `k`, and `g` have shape `[T,H,Dk]`;
`v` is `[T,H,Dv]` and `beta` is `[T,H]`. Q/K are already L2-normalized.
Q/K/V share BF16 or FP16; `g` is FP32 bounded log decay in [-5,0].
All inputs use one CUDA device. `cumulative` gives integer sequence boundaries
and `chunks` gives integer (sequence, chunk) pairs for 64-token chunks; both
are nonempty even for one sequence, unlike the single-sequence sentinels of
`prepare_packed_sequence_metadata`. `Dk` is 32, 64, 128, or 256.
There is zero initial recurrent state and no returned final state.
On SM90 with BF16 Q/K/V and D=128, dispatch prefers
`cula.kimi_delta_attention.sm90.recompute` (native forward, recomputed
intra-chunk matrices in backward), then `cula.kimi_delta_attention.sm90`
(FLA forward, cuLA fused WY backward). Unsupported geometry or missing optional
dependencies uses `fla.kimi_delta_attention.chunk`. Fast arithmetic is the
default; `fla.kimi_delta_attention.chunk.ieee` is an explicit accuracy override.
Autograd returns gradients
for Q, K, V, decay and beta; metadata has no gradient. No caller tensors are
retained in a global cache.

### `kda_decay`

`kda_decay(raw, dt_bias, a_log, *, lower_bound=-5.0)` maps projected
`raw[T,H,D]`, per-channel `dt_bias[H*D]`, and per-head `a_log[H]`
to FP32 negative log decay. All exponential/sigmoid arithmetic uses FP32.
A negative `lower_bound` bounds the result; KDA requires values in [-5,0].
The ordinary PyTorch expression differentiates all inputs.

## Mixture of experts

### `moe`

**Status:** Public

**Signature**

```text
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
    compute_precision: str | None = None,
    swiglu_limit: float | None = None,
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
| `router_weight` | `Tensor[E,D]` | router projection (`nn.Linear` weight), any float dtype; the logits come from `mlops.kernels.router`: operands at `h2`'s dtype (bf16/fp16 on CUDA), sums and logits in fp32, the weight's gradient summed in fp32 from two bf16 pieces of the cotangent and returned at the weight's dtype (or `weight_gradient_dtype`) |
| `w13_experts` | `Tensor[E,2F,D]` | packed gate/up expert weights: per expert an `nn.Linear` weight whose first `F` rows are the gate and last `F` rows the up projection |
| `w2_experts` | `Tensor[E,D,F]` | expert down weights (`nn.Linear` weights) |
| `top_k` | positive `int` | routes selected per row |
| `routing_mode` | `str` | `softmax_then_topk`, `topk_then_softmax`, or `sigmoid_noaux_tc` |
| `router_bias` | `Optional[Tensor[E]]` | optional bias used for selection |
| `n_group` | positive `int` | expert routing group count |
| `topk_group` | positive `int` | groups retained before expert selection |
| `routed_scaling` | `float` | multiplier for selected route weights |
| `lengths` | `Optional[Sequence[int]]` | sequence-wise auxiliary-loss boundaries; omitted means one segment |
| `shared_w13` | `Optional[Tensor[2Fs,D]]` | optional shared gate/up projection (gate rows, then up rows) |
| `shared_w2` | `Optional[Tensor[D,Fs]]` | optional shared down projection |
| `shared_gate_weight` | tensor compatible with `[1,D]` | required by sigmoid-gated shared mode |
| `shared_mode` | `Literal["add", "sigmoid"]` | shared-expert combination |
| `compute_precision` | `Optional[str]` | the precision of the experts' GEMMs, apart from how the weights are stored. `None`: the weights' own -- `"fp8_current"` when `w13_experts`/`w2_experts` are `QuackFP8Weight` containers (`expert_parallel/quack/parameters/fp8.py`: each weight's E4M3 rows and transpose with a descale per row, built once by `QuackFP8Weight.quantize(weight, interleaved=True)` for `w13` and `.quantize(weight)` for `w2`, or published from a master with `copy_`), `"fp8_block"` when they are `BlockFP8Weight` containers (`mlops/block_fp8.py`: E4M3 stored once with an FP32 scale per 128x128 block, rows in `nn.Linear` order, built by `BlockFP8Weight.quantize(weight)` or taken from an FP8 checkpoint's blocks), bf16 otherwise. `"fp8_current"`: E4M3 GEMMs with current per-row scaling of the routed rows and cotangents, served by the SonicMoE implementations on SM90; bf16 weights are quantized per call. `"fp8_block"`: the same rows against the weights' block scales, one stored copy serving both the projections and the input gradients (SonicMoE, SM90); bf16 weights are quantized per call. `"bf16"`: bf16 GEMMs, FP8 containers dequantized per call (every implementation). Weight gradients come at the weights' logical dtype in `nn.Linear` row order; implementations that cannot honor a request decline, and dispatch fails when none can |
| `swiglu_limit` | `Optional[float]`, positive | clamps every expert's SwiGLU, routed and shared, as GLM's experts do: `silu(min(gate, limit)) * clamp(up, -limit, limit)`, with no gradient through a clipped value (`packed_swiglu(limit=)`'s semantics, a value exactly at the limit passing its gradient); `None` leaves SwiGLU unclamped. Every implementation serves it |

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
`router_weight`, `w13_experts`, and `w2_experts`; the builtin and SonicMoE
implementations return the weights' at
[`weight_gradient_dtype`](DISPATCH.md#selection-and-precision) when
one is set. A router or expert weight that does not require a gradient
receives none, and its weight-gradient GEMM is skipped. Router bias, counts,
probability diagnostics, and discrete route metadata do not receive gradients.
The operation itself does not update router-bias or round counters.

**Constraints and exceptions**

- `h2.shape == residual.shape`.
- Every weight is an `nn.Linear` weight, `[out, in]` per expert:
  `router_weight` is `[E,D]`, `w13_experts[e]` is `[2F,D]` with gate rows then
  up rows, and `w2_experts[e]` is `[D,F]`; their `D`, `E`, and `F` must agree.
- `1 <= top_k <= E`.
- Shared W13/W2 must be supplied together.
- `shared_mode` must be `add` or `sigmoid`.
- Sigmoid mode requires `shared_gate_weight`.
- Group counts and `lengths` must be compatible with routing mode.

Invalid shared-expert combinations, an unknown `compute_precision`, and a
nonpositive `swiglu_limit` raise `ValueError`; other incompatible shapes or
implementations raise their backend error.

**Implementations**

`sonicmoe.moe.composed` -- the default on compute capability 9.x and 10.x
when the `sonicmoe` extra is installed -- and its single-region form
`sonicmoe.moe`; `builtin.moe.grouped_gemm_composed` (the default elsewhere)
and the single-region opaque control `builtin.moe.grouped_gemm`; or
`native_torch.moe`. The SonicMoE forms are semantic-only and bfloat16-only.
The composed builtin keeps the same semantic call but exposes routing/dispatch/
`w13` separately from SwiGLU/`w2`/combine so AOT can rematerialize `h13`
without replaying the expert down projection. Routed expert activation remains
part of the selected MoE implementation; it does not consult the standalone
`swiglu` operation.

The manual explicit forward/backward API resolves to the opaque
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

### `sigmoid_topk`

`sigmoid_topk(logits, correction_bias, top_k, *, groups=1,
selected_groups=1, scale=2.5, normalize=True)` returns expert IDs and weights.
`logits[T,E]` must be FP32; `correction_bias[E]` is read in FP32. Selection
can first restrict experts to `selected_groups` of the equal-sized `groups`.
Bias affects selection only; differentiable weights come from unbiased sigmoid
scores, optionally normalized then multiplied by `scale`.
Dense router projection and any balancing objective remain caller-owned.

## Hyper-connections

The semantic boundaries are **before** and **after** the attention/MLP branch.
Parameters belong to the model and are passed as tensors; every tensor input is
differentiable. Providers may fuse within a boundary. The native composition is
visible to AOTAutograd and recomputation partitioning; an opaque fused provider
must register its fake function, VJP, saved tensors and FLOP accounting.

### `mhc_pre`

`mhc_pre(streams, projection, bias, scale, *, eps=1e-5,
sinkhorn_eps=1e-6, iterations=20)` includes flattened RMS normalization,
coefficient projection, pre/post gates, Sinkhorn, and branch-input collapse.
For `streams[...,N,D]`, the projection is `[N*(N+2),N*D]`, bias is `[N*(N+2)]`
and scale is `[3]`. Returns `(post[...,N], mixing[...,N,N], branch_input[...,D])`.
Computation through coefficient construction is FP32; branch input retains the
activation dtype. Parameter storage dtypes are unchanged.

Implementations: `native_torch.mhc_pre`, `tilekernels.mhc_pre.sinkhorn`, and
`deepgemm.mhc_pre`. TileKernels alone fuses Sinkhorn forward/backward;
normalization/projection and collapse remain ordinary tensor code.
DeepGEMM combines its TF32 HC prenorm projection/squared-norm reduction with
TileKernels' fused RMS scaling, gates, Sinkhorn and stream collapse. It keeps
coefficients and accumulators FP32 and checkpoint storage unchanged; TF32
projection and algebraic reordering change floating-point rounding.

The DeepGEMM provider requires SM90/SM100, contiguous BF16 four-stream inputs,
H divisible by 128, T divisible by 32, and BF16/FP32 projection weights. It is
preferred for supported geometry;
`MLOPS_IMPLEMENTATIONS=mhc_pre=deepgemm.mhc_pre` forces it explicitly. The forward saves inputs and small FP32
projection/norm statistics instead of a full FP32 normalized activation.
Backward recomputes gates/Sinkhorn, uses component VJPs and a fused TF32x3 projection/input-gradient
kernel. Neither forward nor backward materializes a full FP32 activation
workspace. Frozen parameters avoid their parameter-gradient GEMMs; trainable
projection gradients use bounded 256-token chunks. Sinkhorn iteration states
are reconstructed on chip. These providers support first-order autograd.

### `mhc_combine`

`mhc_combine(branch, streams, post, mixing, *, coefficient_precision="fp32")`
is the complete post operation: mix residual streams, expand the branch, add.
`mixing[...,i,j]` sends input stream i to output stream j. FP32 coefficients and
accumulation are the default, with final output cast to the activation dtype.
The optimized providers never materialize full FP32 activation copies.
`coefficient_precision="activation"` retains the GLM HF reference's reduced
coefficient operands and intermediate rounding (compiler fusion can affect
rounding of ordinary elementwise code).

Implementations: `native_torch.mhc_combine`,
`builtin.mhc_combine.four_stream`, and optional `tilekernels.mhc_combine`.
TileKernels is preferred for supported FP32-coefficient calls on SM90/SM100;
its SM90 forward/backward are hardware-tested, and SM100 eligibility follows
upstream support without hardware testing. Other architectures, unsupported
shapes, and activation-precision mode fall back to the builtin/native
providers.

```python
post, mixing, branch_input = mhc_pre(streams, fn, bias, scale)
branch = attention_or_mlp(branch_input)
streams = mhc_combine(branch, streams, post, mixing)
```

### `mhc_normalize_streams`

`mhc_normalize_streams(streams, *, eps=1e-5)` flattens `streams[...,N,D]` to
`[...,N*D]` in FP32 and divides each row by its root mean square, with `eps`
added to the mean square and no gain. The result is the input of a
caller-owned coefficient projection. Ordinary PyTorch autograd differentiates
it.

### `mhc_coefficients`

`mhc_coefficients(streams, projected, bias, scale, *, sinkhorn_eps=1e-6,
iterations=20)` consumes externally projected FP32 logits
`projected[...,N*(N+2)]`, with `bias[N*(N+2)]` and `scale[3]` as in `mhc_pre`,
and returns the same `(post, mixing, branch_input)` tuple. Coefficients are
FP32 whatever the bias and scale dtypes; `sinkhorn_eps` stabilizes the pre
gates and the Sinkhorn normalization, and `iterations` counts its rounds.
Logits that are not FP32, mismatched shapes, and `iterations < 1` raise
`ValueError`. `native_torch.mhc_pre` is `mhc_coefficients` applied to
`mhc_normalize_streams(streams, eps=eps)` projected by `projection` in FP32.

Both helpers are available for custom compositions; model code should use the
two semantic boundaries to permit larger fusion. The private
`mlops::mhc_mix_four` and TileKernels Sinkhorn operations are implementation
details, not required model graph boundaries.

## Language-model epilogues

### `cross_entropy`

**Status:** Public

**Signature**

```text
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
gradients. The builtin Triton implementation rejects class weights, so
weighted calls resolve to `native_torch.cross_entropy`.

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

```text
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

```text
head_loss(
    hidden: Tensor,
    head_weight: Tensor,
    targets: Tensor,
    *,
    chunk_size: int | None = None,
    valid_rows: int | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> Tensor
```

**Purpose**

Computes vocabulary projection and next-token cross entropy from already
normalized hidden states with a bounded chunk-local logits footprint: the mean
over the rows, or the sum for a caller that divides by a total of its own,
such as a step's trained tokens across its microbatches.

**Parameters**

| Name | Type/shape | Description |
|---|---|---|
| `hidden` | `Tensor[...,D]`, flattened to `[R,D]` | caller-normalized hidden states |
| `head_weight` | `Tensor[V,D]` | vocabulary projection |
| `targets` | integer `Tensor[hidden.shape[:-1]]` | next-token IDs |
| `chunk_size` | positive `Optional[int]` | explicit token rows per logits chunk |
| `valid_rows` | `Optional[int]` | the mean's denominator in `[1,R]`; `None` uses `R`; not with `reduction="sum"` |
| `reduction` | `"mean"` or `"sum"` | divide the summed cross entropy by the rows, or return the sum |

**Returns**

Scalar FP32 cross entropy: the mean over the rows, or the sum. A row whose
target is negative adds nothing to the sum and, under `"mean"` without
`valid_rows`, is still one of the rows divided by.

**Autograd and effects**

The chunked forward never retains `[R,V]` logits. It computes and privately
saves unit-cotangent `grad_hidden[R,D]` and `grad_head_weight[V,D]`, the
latter summed over chunks at
[`weight_gradient_dtype`](DISPATCH.md#selection-and-precision) when
one is set. Registered backward scales those tensors out-of-place by the incoming loss
cotangent and does not rerun projection or cross entropy. This supports
repeated first-order VJPs but not higher-order differentiation.

**Constraints and exceptions**

Hidden/target row counts must match; targets must lie below `V`, a negative
target being ignored; `chunk_size > 0`; `1 <= valid_rows <= R`; and
`valid_rows` is not given with `reduction="sum"`. Detected chunked-path
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
Ordinary autograd retains the norm input/statistics and executes its VJP
separately. Save-versus-recompute policy belongs to graph lowering rather than
to separate RMS/Layer head APIs.


### `lora_head_loss`

**Status:** Public

**Signature**

```text
lora_head_loss(
    hidden: Tensor, head_weight: Tensor, lora_a: Tensor, lora_b: Tensor,
    targets: Tensor, *, scale: float = 1.0,
    chunk_size: int | None = None, valid_rows: int | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> Tensor
```

**Purpose**

Cross entropy of `X @ W.T + scale * (X @ A.T) @ B.T`, evaluated in row
chunks. This is a separate operation alongside ordinary `head_loss`.
It never builds a dense `B @ A` weight update or retains full-batch logits.

**Parameters**

| Name | Shape / meaning |
|---|---|
| `hidden` | `[..., D]` normalized hidden states |
| `head_weight` | `[V, D]` base weight, frozen by setting `requires_grad=False` |
| `lora_a` | `[rank, D]` input factor |
| `lora_b` | `[V, rank]` output factor; usually initialized to zero |
| `targets` | One integer label per hidden row; negative labels are ignored |
| `scale` | Finite multiplier, conventionally `alpha / rank` |
| `chunk_size` | Positive maximum rows per logits chunk; same default as `head_loss` |
| `valid_rows` | Mean denominator in `[1, rows]`; omitted uses every row |
| `reduction` | `"mean"` or `"sum"`; a sum cannot specify `valid_rows` |

**Returns**

An FP32 scalar. Mean/sum and ignored-label normalization match `head_loss`.

**Autograd and effects**

The base and hidden use the same compute dtype. Factors are cast to that dtype
before the opaque call; ordinary autograd handles the casts back to their
storage dtype. The chunked pass produces only requested unit-cotangent
gradients for hidden, base, A and B. A frozen base has no `[V, D]` gradient
allocation or matrix product. Fully training the base alongside factors is
also supported, including when it is tied to a trainable embedding.

For `H = X @ A.T` and logits cotangent `dZ`, the LoRA VJPs are
`dH = scale * dZ @ B`, `dA = dH.T @ X`,
`dB = scale * dZ.T @ H`, and the input VJP is
`dX = dZ @ W + dH @ A`. Parameter seeds accumulate at
`weight_gradient_dtype` when configured, otherwise at compute dtype.
Backward scales seeds out of place; repeated first-order VJPs are supported.
No `.item()`, host read of tensor values, or explicit device synchronization
is needed. The operation neither changes parameters nor owns optimizer state.

**Constraints and exceptions**

Factors must be two-dimensional with positive matching rank, and all inputs
share a device. Hidden/base/factors are floating point; the base and hidden
dtypes match. Label IDs must be below the vocabulary size. Invalid shapes/dtypes
are rejected by dispatch; invalid scalar policy raises `ValueError`.
Dropout and higher-order derivatives are not part of this operation.

**Implementations**

`builtin.lora_head_loss.chunked` is the default.
`native_torch.lora_head_loss` is the independent full-logits PyTorch reference.

**Example**

```python
head_weight.requires_grad_(False)
loss = lora_head_loss(
    normalized, head_weight, lora_a, lora_b, targets,
    scale=alpha / rank, reduction="sum",
)
```

Call this operation from the head module selected for LoRA. Merely replacing
a Linear's forward does not change callers that read its weight directly.
