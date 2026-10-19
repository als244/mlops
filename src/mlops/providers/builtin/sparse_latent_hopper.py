"""Hopper sparse latent attention: FlashMLA or pipelined TileLang forward, cuDNN DSA backward.

On SM90 the generic TileLang kernels in ``sparse_latent.py`` leave most of the
GPU idle: measured in isolation (docs/plans/sparse_attention_1019) the forward
reaches about 120 effective TFLOPS at 16K tokens and the backward about 35,
against 420 and 360 for the kernels used here. The forward is DeepSeek's
FlashMLA sparse prefill kernel (sm90a, built by scripts/setup_flashmla.sh) when
it is installed, else tile-ai/tilelang's warp-specialized
``sparse_mla_fwd_pipelined`` (``kernels/sparse_mla/forward_pipelined.py``, three
marked local fixes); the backward is cuDNN Frontend's DeepSeek-sparse-attention
backward for SM90 (``cudnn.deepseek_sparse_attention``, CuTe DSL). FlashMLA and
cuDNN take the flat packed layout directly: bare 512-wide tensors (576 with
positional tails), global int32 indices with -1 for unused slots, natural-log
LSE. The TileLang forward sees the generic provider's physical layout (a zero
64-wide tail without positional tails, a sentinel index for unused slots). The
stored LSE is kept in the log2 domain, as the generic provider stores it, so
outputs and residuals stay interchangeable. Selected ahead of the generic
provider on SM90 when tilelang and cuDNN's DSA module import; the generic
provider remains the fallback everywhere else.
"""

import math
from importlib.util import find_spec

import torch
from torch.nn import functional as F

from ...dispatch import attention_costs as costs
from ...dispatch.costs import flop_formula
from ...dispatch.registry import Implementation, SupportResult, register_implementation
from ...dsa import _validate_sparse_latent, _validate_sparse_latent_rope

CAPABILITY = 9  # validated on SM90; the cuDNN DSA backward used here is SM90 specific
_LN2 = math.log(2.0)
_LOG2E = 1.0 / _LN2
# The pipelined TileLang forward consumes 64-wide index blocks two at a time;
# FlashMLA's tiles pair up to 512 (the width NeMo and Megatron pad to).
_FORWARD_BLOCK = 128
_FLASHMLA_BLOCK = 512
# The cuDNN backward tiles 64 query heads; GLM-5.3 has 64.
_HEAD_TILE = 64
_LATENT = 512


def _physical(q, kv, indices, q_rope=None, k_rope=None):
    """Forward layout: [1, T, H, 576] query, [1, T + 1, 1, 576] KV, [1, T, 1, W] int32 indices.

    A zero tail keeps the non-positional math; the extra zero KV row makes the
    sentinel index of an unused slot an in-bounds gather (the kernel masks it).
    """
    query = F.pad(q, (0, 64)) if q_rope is None else torch.cat((q, q_rope), -1)
    values = F.pad(kv, (0, 64)) if k_rope is None else torch.cat((kv, k_rope), -1)
    values = F.pad(values, (0, 0, 0, 1))
    query = query.unsqueeze(0).contiguous()
    values = values.unsqueeze(0).unsqueeze(2).contiguous()
    ids = F.pad(indices, (0, (-indices.shape[-1]) % _FORWARD_BLOCK), value=-1)
    ids = torch.where(ids >= 0, ids, q.shape[0]).to(torch.int32)
    return query, values, ids[None, :, None, :].contiguous()


def _flat(q, kv, indices, block, q_rope=None, k_rope=None):
    """FlashMLA and cuDNN layout: bare [T, H, D] query, [T, D] KV, [T, W] int32 indices, -1 unused."""
    query = q if q_rope is None else torch.cat((q, q_rope), -1)
    values = kv if k_rope is None else torch.cat((kv, k_rope), -1)
    ids = F.pad(indices, (0, (-indices.shape[-1]) % block), value=-1).to(torch.int32)
    return query.contiguous(), values.contiguous(), ids.contiguous()


_flashmla_state: dict = {}


def _flashmla():
    """FlashMLA's sparse forward and the factor taking its LSE into the log2 domain, or None.

    FlashMLA builds differ in the base of the LSE they return: the main-line
    sparse prefill kernel documents a base-2 LSE, NVIDIA's training branch
    returns a natural-log one. The base is measured once per process on a tiny
    problem against a PyTorch reference rather than assumed from the pin.
    """
    if "entry" in _flashmla_state:
        return _flashmla_state["entry"]
    if find_spec("flash_mla") is None or find_spec("flash_mla.cuda") is None:
        _flashmla_state["entry"] = None
        return None
    from flash_mla import flash_mla_sparse_fwd

    with torch.no_grad():
        tokens, heads, width = 8, _HEAD_TILE, _LATENT
        generator = torch.Generator(device="cuda").manual_seed(0)
        q = torch.randn(tokens, heads, width, device="cuda", dtype=torch.bfloat16, generator=generator)
        kv = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16, generator=generator)
        ids = torch.arange(tokens, device="cuda", dtype=torch.int32)[None, :].expand(tokens, tokens)
        ids = torch.where(ids <= torch.arange(tokens, device="cuda", dtype=torch.int32)[:, None], ids, -1)
        ids = F.pad(ids, (0, _FLASHMLA_BLOCK - tokens), value=-1).contiguous()
        _out, _max, lse = flash_mla_sparse_fwd(q, kv[:, None, :], ids[:, None, :], 0.125, d_v=512)
        logits = torch.einsum("thd,sd->ths", q.float(), kv.float()) * 0.125
        logits = logits.masked_fill(~(ids[:, None, :tokens] >= 0), float("-inf"))
        natural = torch.logsumexp(logits, -1)
        if torch.allclose(lse, natural, atol=2e-2, rtol=1e-3):
            factor = _LOG2E
        elif torch.allclose(lse, natural * _LOG2E, atol=2e-2, rtol=1e-3):
            factor = 1.0
        else:
            raise RuntimeError("flash_mla_sparse_fwd returned an LSE in neither natural-log nor log2 form")
    _flashmla_state["entry"] = (flash_mla_sparse_fwd, factor)
    return _flashmla_state["entry"]


def _run_forward(q, kv, indices, scale, q_rope=None, k_rope=None):
    if (flashmla := _flashmla()) is not None:
        sparse_fwd, lse_to_log2 = flashmla
        query, values, ids = _flat(q, kv, indices, _FLASHMLA_BLOCK, q_rope, k_rope)
        output, _max_logits, lse = sparse_fwd(query, values[:, None, :], ids[:, None, :], float(scale), d_v=512)
        return output, lse * lse_to_log2 if lse_to_log2 != 1.0 else lse

    from ...kernels.sparse_mla.forward_pipelined import sparse_mla_fwd

    query, values, ids = _physical(q, kv, indices, q_rope, k_rope)
    q_start = torch.zeros(1, dtype=torch.int32, device=q.device)
    output, lse = sparse_mla_fwd(
        query, values, ids, q_start, q.shape[1], q.shape[-1], 64, ids.shape[-1], 1, 1, scale, True, True
    )
    return output.squeeze(0), lse.squeeze(0)


def _run_backward(dy, q, kv, indices, output, lse, scale, q_rope=None, k_rope=None):
    from cudnn.deepseek_sparse_attention import sparse_attention_backward_wrapper

    # Unused slots stay -1: the kernel ignores every index outside [0, T).
    query, values, ids = _flat(q, kv, indices, 64, q_rope, k_rope)
    sink = torch.full((q.shape[1],), float("-inf"), dtype=torch.float32, device=q.device)
    result = sparse_attention_backward_wrapper(
        query,
        values,
        output.contiguous(),
        dy.contiguous(),
        (lse * _LN2).contiguous(),  # the forward stores a log2-domain LSE
        sink,
        ids,
        softmax_scale=float(scale),
    )
    return result["dq"], result["dkv"]


@torch.library.custom_op("mlops::sparse_latent_attention_hopper_fwd", mutates_args=())
def _forward(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_forward(q, kv, indices, scale)


@_forward.register_fake
def _fake(q, kv, indices, scale):
    return torch.empty_like(q), q.new_empty(q.shape[:2], dtype=torch.float32)


@torch.library.custom_op("mlops::sparse_latent_attention_hopper_bwd", mutates_args=())
def _backward(
    dy: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dq, dkv = _run_backward(dy, q, kv, indices, output, lse, scale)
    return dq.contiguous(), dkv.contiguous()


@_backward.register_fake
def _fake_bwd(dy, q, kv, indices, output, lse, scale):
    return torch.empty_like(q), torch.empty_like(kv)


def _setup(ctx, inputs, output):
    q, kv, indices, ctx.scale = inputs
    result, lse = output
    ctx.save_for_backward(q, kv, indices, result, lse)
    ctx.mark_non_differentiable(lse)


def _autograd(ctx, dy, _):
    return (*_backward(dy, *ctx.saved_tensors, ctx.scale), None, None)


_forward.register_autograd(_autograd, setup_context=_setup)


@torch.library.custom_op("mlops::sparse_latent_rope_hopper_fwd", mutates_args=())
def _rope_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_forward(q, kv, indices, scale, q_rope, k_rope)


@_rope_forward.register_fake
def _rope_fake(q, kv, q_rope, k_rope, indices, scale):
    return q.new_empty(q.shape), q.new_empty(q.shape[:2], dtype=torch.float32)


@torch.library.custom_op("mlops::sparse_latent_rope_hopper_bwd", mutates_args=())
def _rope_backward(
    dy: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dkv = _run_backward(dy, q, kv, indices, output, lse, scale, q_rope, k_rope)
    width = q.shape[-1]
    return (
        dq[..., :width].contiguous(),
        dkv[..., :width].contiguous(),
        dq[..., width:].contiguous(),
        dkv[..., width:].contiguous(),
    )


@_rope_backward.register_fake
def _rope_fake_bwd(dy, q, kv, q_rope, k_rope, indices, output, lse, scale):
    return tuple(x.new_empty(x.shape) for x in (q, kv, q_rope, k_rope))


def _rope_setup(ctx, inputs, output):
    q, kv, q_rope, k_rope, indices, ctx.scale = inputs
    result, lse = output
    ctx.save_for_backward(q, kv, q_rope, k_rope, indices, result, lse)
    ctx.mark_non_differentiable(lse)


def _rope_autograd(ctx, dy, _):
    q, kv, q_rope, k_rope, indices, result, lse = ctx.saved_tensors
    dq, dkv, dq_rope, dk_rope = _rope_backward(dy, q, kv, q_rope, k_rope, indices, result, lse, ctx.scale)
    return dq, dkv, dq_rope, dk_rope, None, None


_rope_forward.register_autograd(_rope_autograd, setup_context=_rope_setup)


def _hopper_support(q, kv, indices):
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] != CAPABILITY:
        return SupportResult.no("The pipelined forward and cuDNN DSA backward are validated on SM90 only")
    if q.shape[-1] != _LATENT:
        return SupportResult.no("The cuDNN DSA backward takes a 512-wide latent")
    if q.shape[1] % _HEAD_TILE:
        return SupportResult.no("The Hopper kernels tile 64 query heads per block")
    if find_spec("tilelang") is None:
        return SupportResult.no("Sparse latent attention requires mlops[tilelang]")
    if find_spec("cudnn") is None or find_spec("cudnn.deepseek_sparse_attention") is None:
        return SupportResult.no("The Hopper backward requires nvidia-cudnn-frontend[cutedsl] with its DSA module")
    return SupportResult.yes()


def _supports(q, kv, indices, *, scale, surface, q_rope=None, k_rope=None, **_kwargs):
    if q_rope is not None or k_rope is not None:
        return SupportResult.no("Use the positional-tail implementation for RoPE")
    try:
        _validate_sparse_latent(q, kv, indices)
    except ValueError as error:
        return SupportResult.no(str(error))
    return _hopper_support(q, kv, indices)


def _rope_supports(q, kv, indices, *, scale, surface, q_rope=None, k_rope=None, **_kwargs):
    try:
        _validate_sparse_latent_rope(q, kv, indices, q_rope, k_rope)
    except ValueError as error:
        return SupportResult.no(str(error))
    return _hopper_support(q, kv, indices)


def forward(q, kv, indices, *, scale):
    return _forward(q.contiguous(), kv.contiguous(), indices.contiguous(), float(scale))


def backward(dy, q, kv, indices, output, lse, *, scale):
    return _backward(dy, q, kv, indices, output, lse, float(scale))


def apply(q, kv, indices, *, scale):
    return forward(q, kv, indices, scale=scale)[0]


def rope_forward(q, kv, indices, *, scale, q_rope, k_rope):
    return _rope_forward(
        q.contiguous(), kv.contiguous(), q_rope.contiguous(), k_rope.contiguous(), indices.contiguous(), float(scale)
    )


def rope_backward(dy, q, kv, indices, output, lse, *, scale, q_rope, k_rope):
    return _rope_backward(dy, q, kv, q_rope, k_rope, indices, output, lse, float(scale))


def rope_apply(q, kv, indices, *, scale, q_rope, k_rope):
    return rope_forward(q, kv, indices, scale=scale, q_rope=q_rope, k_rope=k_rope)[0]


@flop_formula(_forward)
def _forward_flops(q, kv, indices, scale, **_kwargs):
    return costs.sparse_latent_attention(q, kv, indices, scale=scale).logical_flops


@flop_formula(_backward)
def _backward_flops(dy, q, kv, indices, output, lse, scale, **_kwargs):
    return costs.sparse_latent_attention(q, kv, indices, scale=scale, entrypoint="backward").logical_flops


@flop_formula(_rope_forward)
def _rope_forward_flops(q, kv, q_rope, k_rope, indices, scale, **_kwargs):
    return costs.sparse_latent_attention(q, kv, indices, scale=scale, q_rope=q_rope, k_rope=k_rope).logical_flops


@flop_formula(_rope_backward)
def _rope_backward_flops(dy, q, kv, q_rope, k_rope, indices, output, lse, scale, **_kwargs):
    return costs.sparse_latent_attention(
        q, kv, indices, scale=scale, q_rope=q_rope, k_rope=k_rope, entrypoint="backward"
    ).logical_flops


IMPLEMENTATION = register_implementation(
    Implementation(
        "sparse_latent_attention",
        "builtin.sparse_latent_attention.hopper",
        "builtin",
        110,
        False,
        _supports,
        apply,
        forward,
        backward,
    )
)

ROPE_IMPLEMENTATION = register_implementation(
    Implementation(
        "sparse_latent_attention",
        "builtin.sparse_latent_attention.rope.hopper",
        "builtin",
        110,
        False,
        _rope_supports,
        rope_apply,
        rope_forward,
        rope_backward,
    )
)
