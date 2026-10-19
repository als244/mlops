# Adapted from tile-ai/tilelang, commit 194c1b897aa5269e9a89d44c88efdf578a3df620,
# examples/deepseek_v32/sparse_mla_fwd_pipelined.py. Copyright TileLang
# contributors. MIT license: TILELANG_LICENSE.
# Kernel arithmetic retained. Three changes, each marked PATCH: a masked row
# gathers row 0 instead of leaving stale shared memory under its zero
# probability (which could be NaN), the log2-domain LSE is stored (upstream
# leaves it uninitialized), and a zero tail width is accepted.
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def sparse_mla_fwd(
    Q,
    KV,
    Indices,
    q_start_index_s,
    heads,
    dim,
    tail_dim,
    topk,
    kv_stride,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=384,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == 0 or tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"  # PATCH: allow no tail
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch, seq_len, seq_len_kv = T.dynamic("batch, seq_len, seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    assert NI % 2 == 0, "NI should be a multiple of 2"
    assert num_stages == 2, "the manually synchronized pipeline requires two stages"
    D = dim
    D_tail = tail_dim
    KV_stride = kv_stride
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    Q: T.Tensor(q_shape, dtype)  # type: ignore
    KV: T.Tensor(kv_shape, dtype)  # type: ignore
    Indices: T.Tensor(indices_shape, indices_dtype)  # type: ignore
    q_start_index_s: T.Tensor(1, indices_dtype)  # type: ignore
    Output = T.empty(o_shape, dtype)
    Lse = T.empty(lse_shape, accum_dtype)

    with T.Kernel((seq_len - kv_stride + 1 if CP0 else seq_len) * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
        Q_shared = T.alloc_shared([H_per_block, D + D_tail], dtype)
        KV_shared = T.alloc_shared([num_stages, BI, D + D_tail], dtype)
        O_shared = Q_shared
        is_kv_valid = T.alloc_shared([num_stages, BI], "bool", scope="shared")

        acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
        acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
        acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
        S_shared = T.alloc_shared([H_per_block, BI], dtype)
        sumexp = T.alloc_fragment([H_per_block], accum_dtype)
        sum_exp_shared = T.alloc_shared([H_per_block], accum_dtype)
        sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
        alpha_shared = T.alloc_shared([H_per_block], accum_dtype, scope="shared")
        alpha_local = T.alloc_fragment([H_per_block], accum_dtype)
        m_i = T.alloc_fragment([H_per_block], accum_dtype)
        m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
        indices_local = T.alloc_var(indices_dtype)
        safe_local = T.alloc_var(indices_dtype)

        bar_q = T.alloc_barrier(arrive_count=384)
        bar_k_ready = T.alloc_barrier(arrive_count=[128] * num_stages)
        bar_k_free = T.alloc_barrier(arrive_count=[256] * num_stages)
        bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
        bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

        b_i, g_i = by, bz
        s_i = (bx + (KV_stride - 1 if CP0 else 0)) if REPLICATE_H == 1 else (bx // REPLICATE_H + (KV_stride - 1 if CP0 else 0))
        q_i = q_start_index_s[0] + s_i
        max_kv_i = (q_i + 1 - KV_stride) // KV_stride

        H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
        H1 = H0 + H_per_block

        tx = T.get_thread_binding()

        T.tma_copy(Q[b_i, s_i, H0:H1, :], Q_shared, barrier=bar_q)
        T.barrier_arrive(bar_q)

        if tx < 128:
            T.set_max_nreg(240, 1)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
            T.fill(acc_o_l, 0)
            T.barrier_wait(bar_q, 0)

            for i_i in T.serial(T.ceildiv(NI, num_stages)):
                for stage in T.unroll(num_stages):
                    T.barrier_wait(bar_k_ready[stage], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[stage, bi_i], 0, -T.infinity(acc_s.dtype))
                    T.wgmma_gemm(Q_shared[:, : D // 2], KV_shared[stage, :, : D // 2], acc_s, transpose_B=True)
                    T.wgmma_gemm(Q_shared[:, D // 2 : D], KV_shared[stage, :, D // 2 : D], acc_s, transpose_B=True)
                    if D_tail > 0:  # PATCH: no tail GEMM for the bare 512-wide layout
                        T.wgmma_gemm(Q_shared[:, D:], KV_shared[stage, :, D:], acc_s, transpose_B=True)

                    T.wait_wgmma(0)

                    if i_i != 0 or stage != 0:
                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * num_stages + stage) & 1) ^ 1)

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared[stage, :, : D // 2], acc_o_l)

                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_arrive(bar_k_free[stage])

            # Rescale
            for h_i in T.Parallel(H_per_block):
                sum_exp_shared[h_i] = sumexp[h_i]
            for h_i, d_i in T.Parallel(H_per_block, D // 2):
                acc_o_l[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
            # PATCH: store the log2-domain LSE the backward kernel consumes.
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])
            T.copy(acc_o_l, O_shared[:, : D // 2])
            T.copy(O_shared[:, : D // 2], Output[b_i, s_i, H0:H1, 0 : D // 2])

        elif tx >= 128 and tx < 256:
            T.set_max_nreg(168, 1)
            T.fill(acc_o_r, 0)
            for i_i in T.serial(T.ceildiv(NI, num_stages)):
                for stage in T.unroll(num_stages):
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * num_stages + stage) & 1))
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared[stage, :, D // 2 : D], acc_o_r)
                    T.barrier_arrive(bar_k_free[stage])
                    if stage != num_stages - 1 or i_i != T.ceildiv(NI, num_stages) - 1:
                        T.barrier_arrive(bar_sScale_and_sS_free)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D // 2):
                acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

            T.copy(acc_o_r, O_shared[:, D // 2 : D])
            T.copy(O_shared[:, D // 2 : D], Output[b_i, s_i, H0:H1, D // 2 : D])
        elif tx >= 256:
            # producer
            T.set_max_nreg(80, 0)
            for i_i in T.serial(T.ceildiv(NI, num_stages)):
                for stage in T.unroll(num_stages):
                    T.barrier_wait(bar_k_free[stage], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local = Indices[
                            b_i,
                            s_i,
                            g_i,
                            (i_i * num_stages + stage) * BI + r * 16 + (tx - 256) // 8,
                        ]
                        is_kv_valid[stage, r * 16 + (tx - 256) // 8] = indices_local <= max_kv_i
                        # PATCH: a masked row gathers row 0 (finite values behind a zero
                        # probability) instead of leaving stale shared memory, which can
                        # hold NaN bit patterns and poison the PV product.
                        safe_local = T.if_then_else(indices_local <= max_kv_i, indices_local, 0)
                        for u in T.serial((D + D_tail) // 64):
                            T.ptx_cp_async(
                                T.access_ptr(
                                    KV_shared[
                                        stage,
                                        r * 16 + (tx - 256) // 8,
                                        64 * u + (tx - 256) % 8 * 8,
                                    ],
                                    "w",
                                    8,
                                ),
                                T.access_ptr(KV[b_i, safe_local, g_i, 64 * u + (tx - 256) % 8 * 8], "r", 8),
                                8,
                            )
                    T.cp_async_barrier_noinc(bar_k_ready[stage])

    return Output, Lse
