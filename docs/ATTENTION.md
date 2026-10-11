# Attention, routing and residual-mixing operations

These general tensor operations are exported from `mlops` and use ordinary
PyTorch autograd. CUDA work uses the current PyTorch stream. The caller supplies
tensors and sequence metadata; it owns model composition, memory placement and
task scheduling.

## Contents

- [Installation and usage](#installation-and-usage)
- [Routing and expert activation](#routing-and-expert-activation)
- [Hyper-connections](#hyper-connections)
- [KDA](#kda)
- [Pooled indexing](#pooled-indexing)
- [Sparse attention and MLA](#sparse-attention-and-mla)
- [Precision](#precision)
- [Validation and scope](#validation-and-scope)
- [Source attribution](#source-attribution)

## Installation and usage

```bash
python -m pip install -e '.[fla,tilelang]'
python -m pytest -ra tests/operations
```

The optional extras pin the tested FLA/fla-core 0.5.2 and TileLang 0.1.14
interfaces. The package otherwise uses the normal MLOps PyTorch/Triton
environment. Importing `mlops` or `mlops.bootstrap` does not load FLA or TileLang.
FLA and TileLang load only when their kernels are called.

```python
import torch
from mlops import packed_swiglu, sigmoid_topk

x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
activation = torch.compile(packed_swiglu, fullgraph=True)(x, limit=10.0)
activation.backward(torch.randn_like(activation))

logits = torch.randn(32, 64, device="cuda", dtype=torch.float32)
bias = torch.zeros(64, device="cuda", dtype=torch.float32)
expert_ids, expert_weights = sigmoid_topk(logits, bias, top_k=8)
```

Model composition, learned dense/LoRA projections, checkpoint mapping, and
training engines remain outside these operations. Functions are organized by
math, not by a model family.

| Module | Public operations |
|---|---|
| `kda.py` | `kimi_delta_attention`, `kda_decay` |
| `dsa.py`, `dsa_indexer.py` | `sparse_latent_attention`, `pooled_topk` alongside existing DSA APIs |
| `mla.py` | `sparse_mla` alongside existing `mla_attention` |
| `hyper_connection.py` | `mhc_normalize_streams`, `mhc_coefficients`, `mhc_combine` |
| `routing.py` | `sigmoid_topk` |
| Existing activation/normalization APIs | `swiglu(..., limit=...)`, `packed_swiglu(..., limit=...)`, `gated_rms_norm(..., activation="sigmoid")` |

KDA, tied-latent sparse attention, and pooled selection use the normal
implementation registry. Their dispatcher operators live under `torch.ops.mlops`;
import `mlops.bootstrap` before loading exported artifacts in a fresh process.
Differentiable atomic attention operations also expose
`mlops.explicit.kimi_delta_attention` and
`mlops.explicit.sparse_latent_attention` for caller-managed forward residuals and
backward calls. Pooled top-k is non-differentiable; the remaining functions compose
ordinary differentiable PyTorch operations.

See [OPS.md](OPS.md) for signatures and [ARCHITECTURE.md](ARCHITECTURE.md)
for the separation between semantic functions, registrations and kernels.

## Routing and expert activation

`sigmoid_topk(logits, correction_bias, top_k, *, groups=1,
selected_groups=1, scale=2.5, normalize=True)` returns selected IDs and
combination weights. The caller supplies FP32 logits from an FP32 projection.
Correction bias changes selection only. Weight gradients
flow through unbiased sigmoid scores; discrete selection is detached.

`swiglu(gate, up, *, limit=None)` and
`packed_swiglu(packed, *, limit=None)` use the existing activation APIs.
With a finite positive limit, the gate is clipped above that limit and the up
projection is clipped to `[-limit, limit]`. The default `None` leaves both
unchanged. Clipping and its gradient masks run within the existing Triton kernels;
there is no separate clipped operation or full-size temporary. Both semantic and
explicit calls accept the option. Packed inputs support arbitrary leading dimensions.

## Hyper-connections

`mhc_normalize_streams(streams)` returns FP32 normalized flattened
streams. The caller applies its dense projection (or LoRA projection) in FP32.
`mhc_coefficients(streams, projected, bias, scale, ...)`
takes this projected FP32 tensor together with `[..., N, D]` streams and returns post coefficients, the doubly
normalized mixing matrix, and the collapsed `[...,D]` branch input.
`mhc_combine(branch, streams, post, mixing)` expands the branch and mixes the
residual streams. These are ordinary differentiable PyTorch expressions;
Inductor can compile them.

## KDA

`kimi_delta_attention(q, k, v, g, beta, cumulative, chunks)` uses packed
`[T,H,D]` Q/K/V. Q/K are already L2-normalized. `g` contains FP32 per-channel
log decay; `beta` contains per-head input gates. The model computes these
from its projections through `kda_decay` and sigmoid.

Metadata is caller-owned CUDA data:

- `cumulative`: integer sequence boundaries, including 0 and T.
- `chunks`: two integer columns `(sequence_index, chunk_index_in_sequence)`
  for chunks of 64 tokens, ordered by sequence and then chunk.
- Unlike MLOps' single-sequence empty sentinel, this operation
  always receives explicit boundaries/chunks, including for one sequence.

The implemented scope is equal query/value head counts, zero initial recurrent
state, no returned final state, and bounded negative decay. Q/K/V use
BF16 or FP16. BF16 is the full-block validated recipe. More permissive FLA
features are not silently advertised by this wrapper.

`gated_rms_norm(x, gate, weight, eps=..., activation="sigmoid")` uses
the existing gated normalization API and FLA fused forward/backward. The default
`activation="silu"` preserves existing callers. Both choices compute the norm
and gate in FP32, then cast the result to the input dtype.

## Pooled indexing

`pooled_topk(q, key, gates, head_weights, position_bias,
boundaries, *, top_k=2048, query_chunk=64, key_chunk=128)` returns fixed-size
INT32 `[T,top_k+pool_size-1]` selected indices, padded with -1.

- Payloads are CUDA tensors; `boundaries` is small, explicit CPU metadata.
- CPU metadata is a runtime input to the custom op; changing its contents does
  not recompile the tested graph.
- Scores are tiled per sequence. No global `[total_tokens,total_tokens]`
  score matrix is allocated.
- If a sequence fits within the key budget, every causal key is selected
  without score computation.
- Tied top-k scores may select different equally ranked pools from
  Transformers' `torch.topk`. These are valid ties, but different selected
  keys can change attention outputs. Bitwise checkpoint parity is not claimed.
- The current long-sequence ranking path uses PyTorch CUDA operations inside
  the custom op. It is bounded but not yet a fused high-throughput indexer.

Selection is non-differentiable, matching the published implementation. Any
separate indexer-training objective would need its own explicitly defined path.

## Sparse attention and MLA

`sparse_latent_attention(q, kv, indices, *, scale)` expects
BF16 `q[T,H,L]`, shared `kv[T,L]`, and integer `indices[T,K]`.
Heads must be divisible by 16; supported latent widths are 64/128/256/512.
Each row must contain at least one valid key. Valid indices must be unique,
causal, and confined to the query's sequence; padding is -1. The indexer
provides these content guarantees. Static shape/dtype errors fail at the
entrypoint; inspecting GPU index contents would require additional validation.

`sparse_mla(query, latent, key_projection, value_projection, indices,
*, scale)` composes MLA projection absorption with the sparse core. Projections remain
caller-owned tensors and can be supplied by ordinary linear layers or LoRA. Projection shapes are
`[H,Dqk,L]` and `[H,Dv,L]`. The attention scale must use the original
query/key width (usually `Dqk**-0.5`), not latent width.

Projection absorption is algebraically equivalent to expanded K/V but changes
intermediate BF16 rounding. Tests compare values and all four input/weight
gradients numerically. Sparse backward accumulates shared-KV contributions
with FP32 atomics; bitwise deterministic results are not promised.

## Precision

| Operation | Arithmetic and output precision |
|---|---|
| Sigmoid router | Requires FP32 logits and correction bias; scores and combination weights stay FP32 |
| KDA inputs | BF16/FP16 Q/K/V, FP32 log decay; beta may be BF16, FP16 or FP32 |
| KDA decay transform | Casts projected gates and decay parameters to FP32 for exponential and sigmoid arithmetic |
| Gated RMSNorm | FP32 normalization and SiLU/sigmoid gate; output casts to the input dtype |
| mHC coefficients | Requires an FP32 coefficient projection; normalization and Sinkhorn iterations use FP32 |
| mHC residual mixing | Collapse accumulates in FP32; final residual mixing casts coefficients to the streams' dtype |
| Pooled indexer | FP32 pooling softmax and scoring; pooled keys retain the key dtype |
| Sparse latent attention | BF16 Q/KV; FP32 logits, softmax, output accumulation, LSE and shared-KV gradient accumulation |
| SwiGLU with clipping | Preserves activation-dtype rounding at clamp/SiLU boundaries; fused kernels use FP32 intermediates |

FP32 router/mHC projections must be computed in FP32, with autocast disabled
around the caller's linear operation. Casting an already rounded BF16 projection
does not recover that accuracy. The operations accept explicit parameters;
the caller owns their storage dtypes. Upstream convolution/projection weights
are not parameters of the attention operations. A model's checkpoint dtype
policy belongs in its model definition and loader.
See `tests/operations/test_precision.py` and `tests/operations/test_kda_block.py`
for full-training and external-LoRA composition.

KDA uses FLA's chunked training algorithm with `safe_gate=True` in both
directions. The caller supplies log-decays bounded in [-5, 0]. The retained
intra-chunk FLA kernels explicitly use IEEE FP32 dot arithmetic, avoiding
excessive reduced decay-gradient error from implicit TF32 arithmetic.
State-kernel launch metadata is computed per call instead of using FLA's
tensor-identity cache. This avoids retaining caller-owned views or reusing metadata
after their storage contents change; the same FLA numerical kernels are used.
Offset construction stays on the current CUDA stream, with no host read or
synchronization. Tables with up to 1,024 boundary entries use one small scan
kernel; larger tables use ordinary asynchronous CUDA tensor operations to bound
the scan's per-block resource usage. Only shapes and strides are inspected on the
host, so changing sequence lengths in the same storage does not return stale offsets.
Chunk-boundary states still use activation precision. Sparse attention uses
FP32 atomics for shared-KV gradients. Neither path promises bitwise identity
to an all-FP32 oracle or bitwise deterministic gradients.

## Validation and scope

Tests run offline against a checked-in, SHA256-verified Hugging Face source
snapshot; they do not download weights or depend on `experimental/`.
GPU checks are gated on CUDA/SM80+ and the particular optional dependency.
The validated GPU is an RTX 5090 (SM120), using PyTorch 2.13.0+cu130,
Triton 3.7.1, FLA/fla-core 0.5.2 and TileLang 0.1.14.

Checks cover eager/compiled outputs and gradients, operation/fake contracts,
BF16/FP16 activation cases, FP32 arithmetic under autocast, changing runtime
sequence metadata, NoPE projection absorption, mHC and full/LoRA KDA blocks.
The reference recurrence uses FP32, so tests use explicit numerical tolerances.

These are reusable operations. Whole-model checkpoint loading, model
composition, SequentialMoE integration and planned end-to-end training are
separate work. The pooled indexer's long-sequence implementation is bounded
but not yet fused/tuned. Exact top-k tie identity and removal of the sparse
kernel's zero positional tail remain optimization/compatibility work.
Quantized attention projections and an indexer-training objective are not
implemented by these APIs.

## Source attribution

- FLA 0.5.2 supplies the KDA training algorithm. Adapted intra-chunk kernels and
  the MIT license live in `src/mlops/kernels/kda/`.
- TileLang sparse MLA forward/backward derive from commit
  `194c1b897aa5269e9a89d44c88efdf578a3df620`, with its MIT license retained in `src/mlops/kernels/sparse_mla/`.
- The independent test reference is Transformers commit
  `536ecc007387a50e77603bb5d92100e9b07514cc`, retained with its Apache-2.0
  license in `tests/operations/references/`.

Exact URLs and source hashes are recorded in
the `provenance.json` files in those two kernel directories and `tests/operations/references/manifest.json`.
Installed third-party source files are not patched.
