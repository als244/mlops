# GLM operation APIs

These installed interfaces live under `mlops.glm` and use ordinary PyTorch autograd. CUDA work uses the current PyTorch stream. Dependencies remain
tensor arguments; no ShadowSpill imports, task scheduling, or spill-pool logic
appears in the operations.

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
python -m pip install -e '.[glm]'
python -m pytest -ra tests/glm
```

The optional extra pins the tested FLA/fla-core 0.5.2 and TileLang 0.1.14
interfaces. The package otherwise uses the normal MLOps PyTorch/Triton
environment. Importing `mlops.glm` does not import accelerator libraries.
FLA and TileLang load only when their kernels are called.

```python
import torch
from mlops.glm import clipped_swiglu, route

x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
activation = torch.compile(clipped_swiglu, fullgraph=True)(x, limit=10.0)
activation.backward(torch.randn_like(activation))

logits = torch.randn(32, 64, device="cuda", dtype=torch.float32)
bias = torch.zeros(64, device="cuda", dtype=torch.float32)
expert_ids, expert_weights = route(logits, bias, top_k=8)
```

Each semantic function listed below is also exported directly from
`mlops.glm`. This optional namespace has one implementation per operation;
it does not participate in the top-level operation-selection registry or
provide the `mlops.explicit` surface. Custom operations have fake and
autograd registrations under `torch.ops.mlops_glm`; import
`mlops.glm.bootstrap` before loading exported artifacts in a fresh process.
The implementation imports no models, training engines or experiment files.

## Routing and expert activation

`routing.route(logits, correction_bias, top_k, *, groups=1,
selected_groups=1, scale=2.5, normalize=True)` returns selected IDs and
combination weights. The caller supplies FP32 logits from an FP32 projection.
Correction bias changes selection only. Weight gradients
flow through unbiased sigmoid scores; discrete selection is detached.

`activation.clipped_swiglu(packed_gate_up, limit=10.0)` returns the gated
activation. The packed last dimension contains gate then up. BF16, FP16, and
FP32 have fused forward/backward Triton kernels. It preserves the significant
rounding points of the ordinary PyTorch BF16 expression.

## Hyper-connections

`hyper_connection.normalize_streams(streams)` returns FP32 normalized flattened
streams. The caller applies its dense projection (or LoRA projection) in FP32.
`hyper_connection.coefficients(streams, projected, bias, scale, ...)`
takes this projected FP32 tensor together with `[..., N, D]` streams and returns post coefficients, the doubly
normalized mixing matrix, and the collapsed `[...,D]` branch input.
`combine(branch, streams, post, mixing)` expands the branch and mixes the
residual streams. These are ordinary differentiable PyTorch expressions;
Inductor can compile them.

## KDA

`kda.kimi_delta_attention(q, k, v, g, beta, cumulative, chunks)` uses packed
`[T,H,D]` Q/K/V. Q/K are already L2-normalized. `g` contains FP32 per-channel
log decay; `beta` contains per-head input gates. The model computes these
from its projections through `gates.kda_decay` and sigmoid.

Metadata is caller-owned CUDA data:

- `cumulative`: integer sequence boundaries, including 0 and T.
- `chunks`: two integer columns `(sequence_index, chunk_index_in_sequence)`
  for chunks of 64 tokens, ordered by sequence and then chunk.
- Unlike MLOps' single-sequence empty sentinel, this operation
  always receives explicit boundaries/chunks, including for one sequence.

The implemented scope is equal query/value head counts, zero initial recurrent
state, no returned final state, and GLM's bounded negative decay. Q/K/V use
BF16 or FP16. BF16 is the full-block validated recipe. More permissive FLA
features are not silently advertised by this wrapper.

`gates.gated_rms_norm(x, gate, weight, eps=...)` applies GLM's FP32 RMS
normalization and sigmoid output gate before casting back.

## Pooled indexing

`indexing.pooled_topk(q, key, gates, head_weights, position_bias,
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

`attention.sparse_latent_attention(q, latent_kv, indices, *, scale)` expects
BF16 `q[T,H,L]`, shared `latent_kv[T,L]`, and integer `indices[T,K]`.
Heads must be divisible by 16; supported latent widths are 64/128/256/512.
Each row must contain at least one valid key. Valid indices must be unique,
causal, and confined to the query's sequence; padding is -1. The indexer
provides these content guarantees. Static shape/dtype errors fail at the
entrypoint; inspecting GPU index contents would require additional validation.

`mla.sparse_mla(query, latent, key_projection, value_projection, indices,
*, scale)` composes MLA projection absorption with the sparse core. Projections remain
caller-owned tensors and can be supplied by ordinary linear layers or LoRA. Projection shapes are
`[H,Dqk,L]` and `[H,Dv,L]`. The attention scale must use the original
query/key width (GLM: `256**-0.5`), not latent width.

Projection absorption is algebraically equivalent to expanded K/V but changes
intermediate BF16 rounding. Tests compare values and all four input/weight
gradients numerically. Sparse backward accumulates shared-KV contributions
with FP32 atomics; bitwise deterministic results are not promised.

## Precision

| Operation/state | Required behavior in the BF16 model |
|---|---|
| Router logits | FP32 linear projection, outside the router operation |
| Router scores/weights | FP32 sigmoid, normalization and weighted scores |
| Router correction bias | FP32 storage; affects selection only |
| KDA convolution weights | FP32 storage, per HF's strict keep-in-FP32 list |
| KDA `dt_bias`, `A_log` | FP32 storage and decay arithmetic |
| KDA decay | FP32 exponential, sigmoid and log decay |
| KDA RMS/output gate | FP32 variance, reciprocal square root, norm-weight multiplication, sigmoid; cast output to activation dtype |
| mHC coefficients | FP32 flattened RMS normalization, projection, sigmoid/softmax and Sinkhorn iterations |
| mHC residual mixing | Collapse accumulates in FP32; final mixing follows HF's casts to the residual dtype |
| Pooled indexer | FP32 pooling softmax then key-dtype weighted sum; FP32 scoring and head weights |
| Sparse attention | BF16 Q/KV; FP32 logits, softmax accumulation, output accumulation, LSE and shared-KV gradient accumulation |
| Clipped SwiGLU | Preserve HF's activation-dtype rounding points; fused kernels use FP32 intermediates |

FP32 router/mHC projections must be computed in FP32, with autocast disabled
around the caller's linear operation. Casting an already rounded BF16 projection
does not recover that accuracy. The operations accept explicit parameters;
the caller owns their storage dtypes. See `tests/glm/test_precision.py` and
`test_block.py` for full-training and external-LoRA composition.

KDA uses FLA's chunked training algorithm with `safe_gate=True` in both
directions. The caller supplies log-decays bounded in [-5, 0]. The retained
intra-chunk FLA kernels explicitly use IEEE FP32 dot arithmetic, avoiding
excessive reduced decay-gradient error from implicit TF32 arithmetic.
Chunk-boundary states still use activation precision. Sparse attention uses
FP32 atomics for shared-KV gradients. Neither path promises bitwise identity
to an all-FP32 oracle or bitwise deterministic gradients.

## Validation and scope

Tests run offline against a checked-in, SHA256-verified Hugging Face source
snapshot; they do not download weights or depend on `experimental/`.
GPU checks are gated on CUDA/SM80+ and the particular optional dependency.
The validated GPU is an RTX 5090 (SM120), using PyTorch 2.13.0+cu130,
Triton 3.7.1, FLA/fla-core 0.5.2 and TileLang 0.1.14.

Validation on 2026-10-09 passed all 87 tests in `tests/glm`. The complete
MLOps suite passed 471 tests, with 39 hardware/opt-in skips elsewhere.

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
  the MIT license live in `src/mlops/glm/kernels/`.
- TileLang sparse MLA forward/backward derive from commit
  `194c1b897aa5269e9a89d44c88efdf578a3df620`, with its MIT license retained.
- The independent test reference is Transformers commit
  `536ecc007387a50e77603bb5d92100e9b07514cc`, retained with its Apache-2.0
  license in `tests/glm/references/`.

Exact URLs and source hashes are recorded in
`src/mlops/glm/provenance.json` and `tests/glm/references/manifest.json`.
Installed third-party source files are not patched.
