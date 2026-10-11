# Adapted from tile-ai/tilelang, commit 194c1b897aa5269e9a89d44c88efdf578a3df620.
# Copyright TileLang contributors. MIT license: TILELANG_LICENSE.
# Kernel arithmetic retained; head tile size is configurable for SM120.
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def sparse_mla_fwd(
    Q,
    KV,
    Indices,
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=128,
    head_tile=16,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        f"haven't check padding correctness yet, dim={tail_dim}"
    )
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > head_tile:
        assert head_kv % head_tile == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // head_tile
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else head_tile

    Q: T.Tensor(q_shape, dtype)  # type: ignore
    KV: T.Tensor(kv_shape, dtype)  # type: ignore
    Indices: T.Tensor(indices_shape, indices_dtype)  # type: ignore
    Output = T.empty(o_shape, dtype)
    Lse = T.empty(lse_shape, accum_dtype)

    with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
        bx,
        by,
        bz,
    ):
        Q_shared = T.alloc_shared([H_per_block, D + D_tail], dtype)
        KV_shared = T.alloc_shared([BI, D], dtype)
        K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
        mask = T.alloc_fragment([BI], "bool")

        acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
        acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
        S_shared = T.alloc_shared([H_per_block, BI], dtype)
        sumexp = T.alloc_fragment([H_per_block], accum_dtype)
        sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
        alpha = T.alloc_fragment([H_per_block], accum_dtype)
        m_i = T.alloc_fragment([H_per_block], accum_dtype)
        m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

        T.fill(acc_o, 0)
        T.fill(sumexp, 0)
        T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

        b_i, g_i = by, bz
        s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
        q_i = s_i
        max_kv_i = q_i

        H0 = g_i * padded_H + (
            0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * head_tile
        )
        H1 = H0 + H_per_block

        # TODO: merge the statements when the compiler has better support for non-power-of-2 extents.
        T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared[:, :D])
        T.copy(Q[b_i, s_i, H0:H1, D:], Q_shared[:, D:])

        for i_i in T.Pipelined(NI, num_stages=num_stages):
            for bi_i in T.Parallel(BI):
                mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

            for bi_i, d_i in T.Parallel(BI, D):
                KV_shared[bi_i, d_i] = KV[
                    b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                ]
            for bi_i, d_i in T.Parallel(BI, D_tail):
                K_tail_shared[bi_i, d_i] = KV[
                    b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i
                ]

            for h_i, bi_i in T.Parallel(H_per_block, BI):
                acc_s[h_i, bi_i] = T.if_then_else(
                    mask[bi_i], 0, -T.infinity(acc_s.dtype)
                )
            T.gemm(
                Q_shared[:, :D],
                KV_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.gemm(
                Q_shared[:, D:],
                K_tail_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(m_i, m_i_prev)
            T.reduce_max(acc_s, m_i, dim=1, clear=False)
            for h_i in T.Parallel(H_per_block):
                m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
            for h_i in T.Parallel(H_per_block):
                alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
            for h_i, bi_i in T.Parallel(H_per_block, BI):
                acc_s[h_i, bi_i] = T.exp2(
                    acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                )
            T.reduce_sum(acc_s, sumexp_i, dim=1)
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

            T.copy(acc_s, S_shared)
            T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        # Rescale
        for h_i, d_i in T.Parallel(H_per_block, D):
            acc_o[h_i, d_i] /= sumexp[h_i]
        for h_i in T.Parallel(H_per_block):
            sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

        T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
        T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return Output, Lse
