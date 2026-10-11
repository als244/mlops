"""Shape-only estimates for packed KDA and shared-latent selected attention."""

from .costs import CostHints, register_operation_estimator


def _bytes(*tensors):
    return sum(x.numel() * x.element_size() for x in tensors)


def kimi_delta_attention(
    q, k, v, g, beta, cumulative, chunks, *, entrypoint="forward", **_kwargs
):
    """Chunked-delta work, using the same convention as linear_attention."""
    tokens, heads, key_width = q.shape
    value_width = v.shape[-1]
    work = (
        tokens
        * heads
        * (4 * key_width * value_width + 2 * 64 * (key_width + value_width))
    )
    gate_work = 8 * g.numel()
    inputs = _bytes(q, k, v, g, beta, cumulative, chunks)
    residuals = 4 * g.numel() + 2 * tokens * heads * 64 * q.element_size()
    forward = entrypoint == "forward"
    return CostHints(
        logical_flops=(work if forward else 5 * work // 2) + gate_work,
        logical_bytes_accessed=inputs
        + residuals
        + _bytes(v)
        + (0 if forward else _bytes(q, k, v, g, beta)),
        notes=(
            "shape-only chunked KDA estimate; chunk width 64; FP32 per-channel gates",
            "packing and padding are not inspected; this is not a measured kernel count",
        ),
    )


def sparse_latent_attention(q, kv, indices, *, scale, entrypoint="forward", **_kwargs):
    """Selected QK/PV work with one KV bank, including backward score replay."""
    tokens, heads, width = q.shape
    work = heads * tokens * indices.shape[-1] * (2 * width)
    forward = entrypoint == "forward"
    return CostHints(
        logical_flops=(2 if forward else 5) * work,
        logical_bytes_accessed=_bytes(q, kv, indices)
        + _bytes(q)
        + 4 * tokens * heads
        + (0 if forward else _bytes(q, kv, q)),
        notes=(
            "selected-key width is a bound including masked padding",
            "the shared latent KV gradient is counted once",
        ),
    )


def pooled_topk(
    q,
    key,
    gates,
    head_weights,
    position_bias,
    boundaries,
    *,
    top_k=2048,
    entrypoint="forward",
    **_kwargs,
):
    """Upper bound for score math; selection comparisons are not FLOPs."""
    if entrypoint != "forward":
        return CostHints(notes=("pooled top-k selection has no gradient",))
    tokens, heads, width = q.shape
    pool = position_bias.shape[0]
    pools = tokens // pool
    work = (
        0
        if tokens <= top_k
        else (
            2 * tokens * heads * width * pools
            + 3 * tokens * heads * pools
            + 5 * tokens * width
        )
    )
    return CostHints(
        logical_flops=work,
        logical_bytes_accessed=_bytes(
            q, key, gates, head_weights, position_bias, boundaries
        )
        + 4 * tokens * (top_k + pool - 1),
        notes=(
            "single-sequence upper bound; CPU boundary values are not inspected",
            "top-k comparisons and index construction are not floating-point arithmetic",
        ),
    )


for _name, _function in (
    ("kimi_delta_attention", kimi_delta_attention),
    ("sparse_latent_attention", sparse_latent_attention),
    ("pooled_topk", pooled_topk),
):
    register_operation_estimator(_name, _function)
