"""The logical work of every operation, independent of any implementation.

One estimator per semantic operation, taking the operation's own forward
arguments plus ``entrypoint`` and returning the ``CostHints`` the canonical
estimate is made of: logical FLOPs and minimum tensor traffic. Each is
registered here, so ``estimate_implementation`` has a canonical answer for
every operation, and every provider's custom operator derives its flop
formula from the same arithmetic instead of carrying its own.

Logical means the mathematical work, not what one kernel does: two per
multiply-add of a matrix product, a small constant per element of an
elementwise or normalizing pass, zero for a gather. Work whose extent depends
on values a shape cannot show is bounded from the shapes: packed attention
charges every sequence at ``max_seqlen``, a mixture of experts charges every
assignment as one row of dense expert work. Minimum traffic reads each input
once and writes each output once; what an implementation adds on top of that
is its own ``estimate`` to report.
"""

from __future__ import annotations

import torch

from .costs import CostHints, register_operation_estimator

#: The chunk length the delta-rule kernels tile a sequence into.
LINEAR_ATTENTION_CHUNK = 64


def _bytes(*tensors) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in tensors
        if isinstance(value, torch.Tensor)
    )


def _rows(value: torch.Tensor) -> int:
    return value.numel() // value.shape[-1]


def _entrypoint(entrypoint: str) -> bool:
    if entrypoint not in {"forward", "backward"}:
        raise ValueError("entrypoint must be 'forward' or 'backward'")
    return entrypoint == "forward"


def rms_norm(x, weight, eps=1e-5, *, entrypoint="forward", **_kwargs) -> CostHints:
    """Square, reduce, rsqrt, scale by rstd and by the weight."""
    del eps
    elements = x.numel()
    rows = _rows(x)
    activation = _bytes(x)
    weight_bytes = _bytes(weight)
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=5 * elements + 3 * rows,
            logical_bytes_accessed=2 * activation + weight_bytes,
            notes=(
                "logical FLOPs count square, reduction, scale, rsqrt, "
                "and two multiplies",
                "logical bytes are minimum tensor traffic and exclude saved residuals",
            ),
        )
    return CostHints(
        logical_flops=10 * elements + rows,
        logical_bytes_accessed=3 * activation + 2 * weight_bytes + 4 * rows,
        notes=(
            "backward logical bytes include dy/x/weight/rstd reads "
            "and dx/dweight writes",
        ),
    )


def layer_norm(
    x, weight, bias=None, eps=1e-5, *, entrypoint="forward", **_kwargs
) -> CostHints:
    """Mean, variance, normalize, and the affine map."""
    del eps
    elements = x.numel()
    rows = _rows(x)
    activation = _bytes(x)
    affine = _bytes(weight, bias)
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=8 * elements + 4 * rows,
            logical_bytes_accessed=2 * activation + affine + 8 * rows,
            notes=("mean and rstd are written once per row at fp32",),
        )
    return CostHints(
        logical_flops=12 * elements + 2 * rows,
        logical_bytes_accessed=3 * activation + 2 * affine + 8 * rows,
        notes=(
            "backward reads dy, x, weight, mean and rstd; "
            "writes dx and the affine gradients",
        ),
    )


def l2_norm(x, eps=1e-6, *, entrypoint="forward", **_kwargs) -> CostHints:
    """Square, reduce, rsqrt, scale."""
    del eps
    elements = x.numel()
    rows = _rows(x)
    activation = _bytes(x)
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=3 * elements + rows,
            logical_bytes_accessed=2 * activation + 4 * rows,
        )
    return CostHints(
        logical_flops=5 * elements,
        logical_bytes_accessed=3 * activation + 4 * rows,
        notes=("backward reads dy, the normalized output and rstd; writes dx",),
    )


def gated_rms_norm(
    x, gate, weight, eps=1e-5, *, entrypoint="forward", **_kwargs
) -> CostHints:
    """RMSNorm of ``x`` multiplied by the SiLU of ``gate``."""
    del eps
    elements = x.numel()
    rows = _rows(x)
    activation = _bytes(x)
    weight_bytes = _bytes(weight)
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=10 * elements + 3 * rows,
            logical_bytes_accessed=3 * activation + weight_bytes + 4 * rows,
            notes=("the norm's five per element, the SiLU's four, and one product",),
        )
    return CostHints(
        logical_flops=18 * elements + rows,
        logical_bytes_accessed=5 * activation + 2 * weight_bytes + 4 * rows,
        notes=(
            "backward reads dy, x, gate, weight and rstd; writes dx, dgate, dweight",
        ),
    )


def swiglu(gate, up, *, entrypoint="forward", **_kwargs) -> CostHints:
    """``silu(gate) * up``: sigmoid, two products."""
    elements = gate.numel()
    element = _bytes(gate) // max(elements, 1)
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=5 * elements,
            logical_bytes_accessed=3 * elements * element,
            notes=("the sigmoid counts as three: exp, add, reciprocal",),
        )
    return CostHints(
        logical_flops=10 * elements,
        logical_bytes_accessed=5 * elements * element,
        notes=("backward reads dy, gate and up; writes dgate and dup",),
    )


def packed_swiglu(packed, *, entrypoint="forward", **_kwargs) -> CostHints:
    """``swiglu`` over the two halves of one ``[rows, 2 * width]`` tensor."""
    elements = packed.numel() // 2
    element = packed.element_size()
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=5 * elements,
            logical_bytes_accessed=3 * elements * element,
        )
    return CostHints(
        logical_flops=10 * elements,
        logical_bytes_accessed=5 * elements * element,
    )


def gelu(x, *, entrypoint="forward", **_kwargs) -> CostHints:
    """The tanh approximation: a cubic, a tanh, and the half-sum product."""
    elements = x.numel()
    activation = _bytes(x)
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=8 * elements,
            logical_bytes_accessed=2 * activation,
            notes=("tanh counts as one operation",),
        )
    return CostHints(
        logical_flops=12 * elements,
        logical_bytes_accessed=3 * activation,
    )


def _rotary(x, positions, cosine, sine, rotated, *, entrypoint) -> CostHints:
    rows = positions.numel()
    width = cosine.shape[-1]
    table = 2 * rows * width * cosine.element_size()
    del sine
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=3 * rotated,
            logical_bytes_accessed=2 * _bytes(x) + _bytes(positions) + table,
            notes=(
                "each rotated element costs two products and one sum",
                "table bytes count the rows the positions select, once",
            ),
        )
    return CostHints(
        logical_flops=3 * rotated,
        logical_bytes_accessed=2 * _bytes(x) + _bytes(positions) + table,
        notes=("the inverse rotation costs what the rotation cost",),
    )


def rope(
    x, positions, base, cosine, sine, *, entrypoint="forward", **_kwargs
) -> CostHints:
    """Rotate every channel of every head."""
    del base
    return _rotary(x, positions, cosine, sine, x.numel(), entrypoint=entrypoint)


def partial_rope(
    x, positions, base, rotary_dim, cosine, sine, *, entrypoint="forward", **_kwargs
) -> CostHints:
    """Rotate the first ``rotary_dim`` channels of every head; pass the rest."""
    del base
    rotated = _rows(x) * int(rotary_dim)
    return _rotary(x, positions, cosine, sine, rotated, entrypoint=entrypoint)


def flash_attention(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    *,
    causal=True,
    entrypoint="forward",
    **_kwargs,
) -> CostHints:
    """Scores and their product with the values, per query head.

    The sequence boundaries are data, so every token is charged as if its
    sequence were ``max_seqlen`` long; packed sequences of equal length make
    that exact. A causal mask halves the pairs.
    """
    tokens, heads, width = q.shape
    pairs = tokens * int(max_seqlen)
    lse = heads * tokens * 4
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=(2 if causal else 4) * heads * width * pairs,
            logical_bytes_accessed=_bytes(q, k, v, cu_seqlens) + _bytes(q) + lse,
            notes=("attention pairs are bounded by tokens times max_seqlen",),
        )
    return CostHints(
        logical_flops=(5 if causal else 10) * heads * width * pairs,
        logical_bytes_accessed=(
            _bytes(q, k, v, cu_seqlens) + 2 * _bytes(q) + lse + _bytes(q, k, v)
        ),
        notes=(
            "backward recomputes the scores and forms dv, dp, dq and dk",
            "backward reads dy, q, k, v, the output and lse; writes dq, dk, dv",
        ),
    )


def dsa_attention(
    q, k, v, indices, lengths, *, entrypoint="forward", **_kwargs
) -> CostHints:
    """Attention from each query to the keys its ``indices`` row selects."""
    del lengths
    tokens, heads, width = q.shape
    value_width = v.shape[-1]
    selected = indices.shape[-1]
    per_head = tokens * selected * (width + value_width)
    lse = heads * tokens * 4
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=2 * heads * per_head,
            logical_bytes_accessed=_bytes(q, k, v, indices) + _bytes(v) + lse,
        )
    return CostHints(
        logical_flops=5 * heads * per_head,
        logical_bytes_accessed=(
            _bytes(q, k, v, indices) + 2 * _bytes(v) + lse + _bytes(q, k, v)
        ),
    )


def causal_conv_silu(
    x, weight, cumulative, chunk_indices=None, *, entrypoint="forward", **_kwargs
) -> CostHints:
    """A depthwise causal convolution of width ``weight.shape[-1]``, then SiLU."""
    del cumulative, chunk_indices
    elements = x.numel()
    taps = weight.shape[-1]
    activation = _bytes(x)
    weight_bytes = _bytes(weight)
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=(2 * taps + 5) * elements,
            logical_bytes_accessed=2 * activation + weight_bytes,
            notes=("two per tap, then the SiLU's five",),
        )
    return CostHints(
        logical_flops=(4 * taps + 6) * elements,
        logical_bytes_accessed=3 * activation + 2 * weight_bytes,
    )


def linear_attention(
    q,
    k,
    v,
    beta,
    a,
    a_log,
    dt_bias,
    cumulative,
    chunk_indices,
    *,
    entrypoint="forward",
    **_kwargs,
) -> CostHints:
    """The chunked gated delta rule.

    Per token and value head: the state update and the read from the state,
    each a product of the key and value widths, plus the intra-chunk
    attention over ``LINEAR_ATTENTION_CHUNK`` tokens.
    """
    del cumulative, chunk_indices
    tokens = q.shape[0]
    key_width = q.shape[-1]
    value_heads, value_width = v.shape[-2], v.shape[-1]
    chunk = LINEAR_ATTENTION_CHUNK
    per_token = 4 * key_width * value_width + 2 * chunk * (key_width + value_width)
    work = tokens * value_heads * per_token
    gate = tokens * value_heads * 4
    matrix = tokens * value_heads * chunk * v.element_size()
    inputs = _bytes(q, k, v, beta, a, a_log, dt_bias)
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=work + 8 * tokens * value_heads,
            logical_bytes_accessed=inputs + _bytes(v) + gate + matrix,
            notes=("the per-head gate and the chunk matrix are the saved residuals",),
        )
    return CostHints(
        logical_flops=5 * work // 2 + 8 * tokens * value_heads,
        logical_bytes_accessed=(
            inputs
            + _bytes(v)
            + gate
            + matrix
            + _bytes(q, k, v, beta, a)
            + 8 * a_log.numel()
        ),
    )


def _moe_geometry(h2, router_weight, w13_experts, w2_experts, top_k):
    rows = _rows(h2)
    return {
        "rows": rows,
        "width": h2.shape[-1],
        "experts": router_weight.shape[1],
        "assignments": rows * int(top_k),
        "packed": w13_experts.shape[2],
        "expert_width": w2_experts.shape[1],
    }


def _moe_prepare_flops(rows, width, experts, assignments, packed, *, forward):
    router = 2 * rows * width * experts + 3 * rows * experts
    up = 2 * assignments * width * packed
    if forward:
        return router + up
    return 2 * router - 3 * rows * experts + 2 * up


def _moe_finish_flops(assignments, expert_width, width, *, forward):
    activation = 5 * assignments * expert_width
    down = 2 * assignments * expert_width * width
    combine = 2 * assignments * width
    if forward:
        return activation + down + combine
    return 2 * activation + 2 * down + combine


def moe_prepare(
    h2, router_weight, w13_experts, *, top_k, entrypoint="forward"
) -> CostHints:
    """The routing and up-projection region of ``moe``: logits, top-k, w13."""
    geometry = _moe_geometry(h2, router_weight, w13_experts, w13_experts, top_k)
    flops = _moe_prepare_flops(
        geometry["rows"],
        geometry["width"],
        geometry["experts"],
        geometry["assignments"],
        geometry["packed"],
        forward=_entrypoint(entrypoint),
    )
    return CostHints(logical_flops=flops)


def moe_finish(h13, w2_experts, *, entrypoint="forward") -> CostHints:
    """The activation, down-projection and combine region of ``moe``."""
    flops = _moe_finish_flops(
        h13.shape[0],
        w2_experts.shape[1],
        w2_experts.shape[2],
        forward=_entrypoint(entrypoint),
    )
    return CostHints(logical_flops=flops)


def moe(
    h2,
    residual,
    router_weight,
    w13_experts,
    w2_experts,
    *,
    top_k,
    entrypoint="forward",
    **_kwargs,
) -> CostHints:
    """Router, dense expert work for every assignment, and the combine.

    Every one of the ``rows * top_k`` assignments is charged as one row of
    each expert matrix product, which is exact whatever the routing.
    """
    geometry = _moe_geometry(h2, router_weight, w13_experts, w2_experts, top_k)
    forward = _entrypoint(entrypoint)
    flops = _moe_prepare_flops(
        geometry["rows"],
        geometry["width"],
        geometry["experts"],
        geometry["assignments"],
        geometry["packed"],
        forward=forward,
    ) + _moe_finish_flops(
        geometry["assignments"],
        geometry["expert_width"],
        geometry["width"],
        forward=forward,
    )
    weights = _bytes(router_weight, w13_experts, w2_experts)
    h13 = geometry["assignments"] * geometry["packed"] * h2.element_size()
    if forward:
        return CostHints(
            logical_flops=flops,
            logical_bytes_accessed=_bytes(h2, residual) + weights + _bytes(h2),
            notes=(
                "minimum traffic reads the activations and every expert once and "
                "writes the output; dispatch and h13 traffic is the implementation's",
            ),
        )
    return CostHints(
        logical_flops=flops,
        logical_bytes_accessed=4 * _bytes(h2) + 2 * weights + h13,
        notes=("backward reads dy, h2, the weights and h13; writes every gradient",),
    )


def cross_entropy(
    logits,
    targets,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    *,
    entrypoint="forward",
    **_kwargs,
) -> CostHints:
    """Row-wise log-sum-exp and the gathered target term."""
    del ignore_index, reduction
    rows, vocabulary = logits.shape
    elements = logits.numel()
    logits_bytes = _bytes(logits)
    targets_bytes = _bytes(targets)
    weight_bytes = _bytes(weight)
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=4 * elements + 2 * rows,
            logical_bytes_accessed=(
                logits_bytes + targets_bytes + weight_bytes + 4 * rows
            ),
            notes=(
                f"logical estimate assumes {rows} rows and vocabulary {vocabulary}",
                "comparisons and max reductions are excluded from the FLOP count",
            ),
        )
    return CostHints(
        logical_flops=4 * elements,
        logical_bytes_accessed=(
            2 * logits_bytes + targets_bytes + weight_bytes + 8 * rows
        ),
        notes=(
            "backward bytes include logits, per-row cotangent/LSE, and logits VJP",
        ),
    )


def head_loss(
    hidden,
    head_weight,
    targets,
    *,
    chunk_size=None,
    valid_rows=None,
    need_hidden_grad=True,
    need_head_grad=True,
    entrypoint="forward",
    **_kwargs,
) -> CostHints:
    """Logits, their cross entropy, and both seed-one gradients, in the forward.

    The backward only scales the two gradients the forward already made.
    """
    del chunk_size, valid_rows
    rows = _rows(hidden)
    width = hidden.shape[-1]
    vocabulary = head_weight.shape[0]
    gradients = int(need_hidden_grad) * _bytes(hidden) + int(need_head_grad) * _bytes(head_weight)
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=2 * rows * width * vocabulary * (1 + int(need_hidden_grad) + int(need_head_grad)) + 6 * rows * vocabulary,
            logical_bytes_accessed=(
                _bytes(hidden, head_weight, targets) + gradients + 4
            ),
            notes=(
                "logits and only requested hidden/head gradient products",
                "the logits are never written whole; the chunking is the implementation's",
            ),
        )
    return CostHints(
        logical_flops=2 * (int(need_hidden_grad) * rows * width + int(need_head_grad) * vocabulary * width),
        logical_bytes_accessed=2 * gradients + 4,
        notes=("backward scales the two seed-one gradients by the loss cotangent",),
    )


def lora_head_loss(
    hidden,
    head_weight,
    lora_a,
    lora_b,
    targets,
    *,
    need_hidden_grad=True,
    need_head_grad=False,
    need_lora_a_grad=True,
    need_lora_b_grad=True,
    weight_grad_dtype=None,
    entrypoint="forward",
    **_kwargs,
) -> CostHints:
    """Chunked base/low-rank projection and requested first-order seeds."""
    rows, width, vocab, rank = (
        _rows(hidden),
        hidden.shape[-1],
        head_weight.shape[0],
        lora_a.shape[0],
    )
    needs = (need_hidden_grad, need_head_grad, need_lora_a_grad, need_lora_b_grad)
    tensors = (hidden, head_weight, lora_a, lora_b)
    seed_elements = sum(
        t.numel() for t, needed in zip(tensors, needs, strict=True) if needed
    )
    seed_bytes = sum(
        t.numel()
        * (
            t.element_size()
            if i == 0 or weight_grad_dtype is None
            else torch.empty((), dtype=weight_grad_dtype).element_size()
        )
        for i, (t, needed) in enumerate(zip(tensors, needs, strict=True))
        if needed
    )
    if not _entrypoint(entrypoint):
        return CostHints(
            logical_flops=seed_elements,
            logical_bytes_accessed=2 * seed_bytes + 4,
            notes=("scale only requested seed-one VJPs by the scalar cotangent",),
        )
    base = 2 * rows * width * vocab
    input_factor = 2 * rows * width * rank
    output_factor = 2 * rows * vocab * rank
    flops = base + input_factor + output_factor + 8 * rows * vocab
    flops += int(need_hidden_grad) * (base + input_factor + rows * width)
    flops += int(need_head_grad) * base
    flops += int(need_lora_a_grad) * input_factor
    flops += int(need_lora_b_grad) * (output_factor + vocab * rank)
    flops += int(need_hidden_grad or need_lora_a_grad) * (output_factor + rows * rank)
    return CostHints(
        logical_flops=flops,
        logical_bytes_accessed=_bytes(*tensors, targets) + seed_bytes + 4,
        notes=(
            "base projection plus two low-rank products; logits consumed in chunks",
            "frozen base weights have no dense gradient product or seed storage",
        ),
    )


def embedding(tokens, weight, *, entrypoint="forward", **_kwargs) -> CostHints:
    """A gather of ``weight`` rows; the backward scatters the rows' gradients."""
    rows = tokens.numel()
    width = weight.shape[-1]
    gathered = rows * width * weight.element_size()
    if _entrypoint(entrypoint):
        return CostHints(
            logical_flops=0,
            logical_bytes_accessed=_bytes(tokens) + 2 * gathered,
            notes=("a gather does no arithmetic",),
        )
    return CostHints(
        logical_flops=rows * width,
        logical_bytes_accessed=_bytes(tokens) + gathered + _bytes(weight),
        notes=("backward adds each row's gradient into the table's gradient",),
    )


def adamw(
    parameter, gradient, exp_avg, exp_avg_sq, step, *, entrypoint="forward", **_kwargs
) -> CostHints:
    """One AdamW update: decay, both moments, bias corrections, the step."""
    if not _entrypoint(entrypoint):
        return CostHints(notes=("optimizer update has no differentiable VJP",))
    return CostHints(
        logical_flops=20 * parameter.numel(),
        logical_bytes_accessed=(
            _bytes(parameter, gradient, exp_avg, exp_avg_sq)
            + _bytes(parameter, exp_avg, exp_avg_sq)
            + 2 * step.element_size()
        ),
        notes=("twenty per element: decay, two moments, two corrections, the update",),
    )


for _name, _estimator in (
    ("rms_norm", rms_norm),
    ("layer_norm", layer_norm),
    ("l2_norm", l2_norm),
    ("gated_rms_norm", gated_rms_norm),
    ("swiglu", swiglu),
    ("packed_swiglu", packed_swiglu),
    ("gelu", gelu),
    ("rope", rope),
    ("partial_rope", partial_rope),
    ("flash_attention", flash_attention),
    ("dsa_attention", dsa_attention),
    ("causal_conv_silu", causal_conv_silu),
    ("linear_attention", linear_attention),
    ("moe", moe),
    ("cross_entropy", cross_entropy),
    ("head_loss", head_loss),
    ("lora_head_loss", lora_head_loss),
    ("embedding", embedding),
    ("adamw", adamw),
):
    register_operation_estimator(_name, _estimator)


__all__ = [
    "LINEAR_ATTENTION_CHUNK",
    "adamw",
    "causal_conv_silu",
    "cross_entropy",
    "dsa_attention",
    "embedding",
    "flash_attention",
    "gated_rms_norm",
    "gelu",
    "head_loss",
    "l2_norm",
    "layer_norm",
    "linear_attention",
    "lora_head_loss",
    "moe",
    "moe_finish",
    "moe_prepare",
    "packed_swiglu",
    "partial_rope",
    "rms_norm",
    "rope",
    "swiglu",
]
