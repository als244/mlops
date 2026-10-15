"""The block-scaled FP8 expert GEMMs
(``expert_parallel/quack/experts/block_fp8_gemm.py``) and the row quantizer's
K order for the transposing one (``expert_parallel/quack/kernels/
quantize_rows.py``).

On the CPU: the transposer's index maps, read from the kernel's own CuTe-DSL
copies and layouts (no device needed), the column restore on the kernel's
accumulator fragment, the quantizer's column map and reference, and compiles
of every GEMM variant for SM90. On an SM90 GPU: the transpose kernel by
itself; the input-gradient GEMM, dense and varlen; the projections, plain,
with a LoRA addend, and the up projection with gate/up interleaved and the
SwiGLU in its epilogue; the input gradient with the SwiGLU backward in its
epilogue -- each against float64 products of the dequantized operands.
"""

from __future__ import annotations

import itertools
import pathlib
import re
import subprocess

import pytest
import torch

SM90 = "sm_90a"


def _gemm_module():
    pytest.importorskip("cutlass")
    pytest.importorskip("quack")
    from mlops.expert_parallel.quack.experts import block_fp8_gemm

    return block_fp8_gemm


def _quantizer():
    pytest.importorskip("triton")
    from mlops.expert_parallel.quack.kernels import quantize_rows

    return quantize_rows


def _traced(fn):
    """``fn()`` evaluated inside a CuTe-DSL trace for SM90 (no device)."""
    from cutlass import cute

    result = {}

    @cute.jit
    def probe():
        result["value"] = fn()

    cute.compile(probe, options=f"--gpu-arch {SM90}")
    return result["value"]


def _prmt(lo, hi, selector):
    """``prmt.b32`` on byte lists: result byte j is byte ``selector >> 4j``
    of ``hi:lo``."""
    source = list(lo) + list(hi)
    return [source[(selector >> (4 * j)) & 7] for j in range(4)]


def _logical_column(physical, order):
    """The logical index a physical index of an order within 16 holds."""
    return physical - physical % 16 + order[physical % 16]


# ---- stage 0: index math, on the CPU -------------------------------------------


def _transpose_maps(g):
    """Per thread of the transposing warpgroup, per copy step (one ldmatrix or
    stmatrix .x4) and byte: the tile coordinate ``(n, k)`` each ldmatrix lane
    addresses (``ld_addr``) and each loaded register byte holds (``ld_data``);
    where stmatrix writes each register byte (``st_data``, a physical row of
    the K-major tile) and which row each stmatrix lane addresses
    (``st_addr``). Plus the byte offsets of the addressed rows in the staged
    shared-memory layouts."""
    import cutlass
    from cutlass import cute

    s2r, r2s = g.transpose_copies(cutlass.Float8E4M3FN)
    tile = cute.make_identity_tensor((g.TILE_N, g.TILE_K))
    mn_view, k_view = g.mn_major_view(tile), g.k_major_view(tile)
    mn_layout, k_layout = g._weight_smem_layouts(cutlass.Float8E4M3FN, 1)

    def table(partition):
        partition = cute.group_modes(partition, 1, cute.rank(partition))
        return [
            [partition[b, i] for b in range(cute.size(partition, mode=[0]))]
            for i in range(cute.size(partition, mode=[1]))
        ]

    maps = {"ld_addr": [], "ld_data": [], "st_data": [], "st_addr": []}
    for thread in range(g.TRANSPOSE_THREADS):
        ld, st = s2r.get_slice(thread), r2s.get_slice(thread)
        maps["ld_addr"].append(table(ld.partition_S(mn_view)))
        maps["ld_data"].append(table(ld.partition_D(mn_view)))
        maps["st_data"].append(table(st.partition_S(k_view)))
        maps["st_addr"].append(table(st.partition_D(k_view)))
    maps["ld_offset"] = [
        [cute.crd2idx((*min(row), 0), mn_layout) for row in steps]
        for steps in maps["ld_addr"]
    ]
    maps["st_offset"] = [
        [cute.crd2idx((*min(row), 0), k_layout) for row in steps]
        for steps in maps["st_addr"]
    ]
    return maps


@pytest.fixture(scope="module")
def transpose_maps():
    g = _gemm_module()
    return g, _traced(lambda: _transpose_maps(g))


def test_the_transposer_moves_every_byte_to_the_k_and_n_orders(transpose_maps):
    g, maps = transpose_maps
    moved = {}
    for thread in range(g.TRANSPOSE_THREADS):
        for step, data in enumerate(maps["ld_data"][thread]):
            words = [data[4 * w : 4 * w + 4] for w in range(4)]
            permuted = []
            for pair in range(2):
                lo, hi = words[2 * pair], words[2 * pair + 1]
                permuted += [_prmt(lo, hi, selector) for selector in g.PRMT_SELECTORS]
            data = [byte for word in permuted for byte in word]
            for byte, physical in enumerate(maps["st_data"][thread][step]):
                assert physical not in moved, f"two bytes written to {physical}"
                moved[physical] = data[byte]
    assert len(moved) == g.TILE_N * g.TILE_K
    assert sorted(moved.values()) == sorted(moved)  # a bijection onto the tile
    for (n, k), source in moved.items():
        expected = (_logical_column(n, g.N_ORDER), _logical_column(k, g.K_ORDER))
        assert source == expected, f"row {n}, column {k} holds {source}"


def test_the_transposer_copies_are_free_of_bank_conflicts(transpose_maps):
    g, maps = transpose_maps
    for offsets in (maps["ld_offset"], maps["st_offset"]):
        for warp in range(g.TRANSPOSE_THREADS // 32):
            for step in range(len(offsets[0])):
                for matrix in range(4):  # lanes 8m..8m+7 address matrix m's rows
                    lanes = range(32 * warp + 8 * matrix, 32 * warp + 8 * matrix + 8)
                    rows = [offsets[lane][step] for lane in lanes]
                    assert all(row % 16 == 0 for row in rows)
                    # 16-byte rows: eight distinct 16-byte bank groups.
                    assert len({row % 128 // 16 for row in rows}) == 8


def test_the_two_weight_views_share_bytes():
    """TMA's MN-major view and wgmma's K-major view of the staged weight tile
    are 128-byte swizzles of the same bytes, stage by stage (the swizzles the
    bank-conflict check reads offsets from)."""
    import cutlass
    from cutlass import cute

    g = _gemm_module()
    stages = 3
    # Strides 3 and 5 visit every residue modulo 8 and 16 of both indices.
    points = [
        (n, k, s)
        for s in range(stages)
        for n in range(0, g.TILE_N, 3)
        for k in range(0, g.TILE_K, 5)
    ]

    def offsets():
        mn_layout, k_layout = g._weight_smem_layouts(cutlass.Float8E4M3FN, stages)
        return [(cute.crd2idx(p, mn_layout), cute.crd2idx(p, k_layout)) for p in points]

    tile = g.TILE_N * g.TILE_K
    for (n, k, s), (mn_offset, k_offset) in zip(points, _traced(offsets)):
        assert mn_offset == tile * s + 128 * k + 16 * ((n // 16) ^ (k % 8)) + n % 16
        assert k_offset == tile * s + 128 * n + 16 * ((k // 16) ^ (n % 8)) + k % 16


def test_restoring_the_column_order():
    """On the kernel's own accumulator fragment, every MMA thread's slot
    ``(c, j, g)`` takes its value from a slot of the same thread and row whose
    physical column holds the slot's logical column."""
    import cutlass
    from cutlass import cute

    g = _gemm_module()
    tile = (g.TILE_M, g.TILE_N, g.TILE_K)
    gemm = g.GemmSm90BlockFP8T(cutlass.Float32, cutlass.Float8E4M3FN, tile, (1, 1, 1))
    gemm.b_dtype = cutlass.Float8E4M3FN

    def fragments():
        gemm._setup_tiled_mma()
        coordinates = cute.make_identity_tensor((g.TILE_M, g.TILE_N))
        threads = gemm.mma_warp_groups * gemm.num_threads_per_warp_group
        result = []
        for thread in range(threads):
            acc = gemm.tiled_mma.get_slice(thread).partition_C(coordinates)
            assert cute.size(acc, mode=[1]) == cute.size(acc, mode=[2]) == 1
            pairs, halves, groups = acc.shape[0]
            result.append(
                {
                    (c, j, group): acc[(c, j, group), 0, 0]
                    for c in range(pairs)
                    for j in range(halves)
                    for group in range(groups)
                }
            )
        return result

    slots = _traced(fragments)
    assert len(slots) == 256 and len(slots[0]) == 64
    for thread in slots:
        for (c, j, group), (row, col) in thread.items():
            source_row, source_col = thread[g.column_source(c, j, group)]
            assert source_row == row
            assert _logical_column(source_col, g.N_ORDER) == col


def test_the_orders_are_the_kernels():
    g = _gemm_module()
    quantize_rows = _quantizer()
    assert g.K_ORDER == quantize_rows.TRANSPOSED_GEMM_ORDER
    # N_ORDER swaps bits 0 and 3; both orders are permutations of 0..15.
    assert sorted(g.N_ORDER) == sorted(g.K_ORDER) == list(range(16))
    assert all(g.N_ORDER[p] == ((p & 1) << 3 | (p & 6) | (p >> 3)) for p in range(16))


def test_the_quantizer_column_map():
    quantize_rows = _quantizer()
    order = quantize_rows.TRANSPOSED_GEMM_ORDER
    columns = quantize_rows.source_columns(48, transposed_gemm=True)
    assert columns.tolist() == [_logical_column(c, order) for c in range(48)]
    columns = quantize_rows.source_columns(8, gate_up=True)
    assert columns.tolist() == [0, 2, 4, 6, 1, 3, 5, 7]
    both = quantize_rows.source_columns(64, transposed_gemm=True, gate_up=True)
    logical = quantize_rows.source_columns(64, transposed_gemm=True)
    assert both.tolist() == [2 * c if c < 32 else 2 * (c - 32) + 1 for c in logical]
    with pytest.raises(ValueError):
        quantize_rows.source_columns(40, transposed_gemm=True)
    with pytest.raises(ValueError):
        quantize_rows.source_columns(7, gate_up=True)


@pytest.mark.parametrize("transposed_gemm, gate_up", [(True, False), (True, True)])
def test_the_quantizer_reference_orders_gathers_and_deinterleaves(
    transposed_gemm, gate_up
):
    quantize_rows = _quantizer()
    torch.manual_seed(0)
    value = torch.randn(40, 64) * torch.linspace(0.1, 3, 64)
    gather = torch.randint(0, 40, (25,))
    data, scales = quantize_rows.quantize_rows_fp8_reference(
        value, gather=gather, transposed_gemm=transposed_gemm, gate_up=gate_up
    )
    plain, plain_scales = quantize_rows.quantize_rows_fp8_reference(value[gather])
    columns = quantize_rows.source_columns(
        64, transposed_gemm=transposed_gemm, gate_up=gate_up
    )
    assert torch.equal(scales, plain_scales)
    assert torch.equal(data.view(torch.uint8), plain[:, columns].view(torch.uint8))
    # A row's largest magnitude maps to 448.
    assert torch.allclose(data.float().abs().amax(dim=1), torch.full((25,), 448.0))


def _nvdisasm():
    path = pathlib.Path(torch.__file__).parents[1] / "nvidia/cu13/bin/nvdisasm"
    return path if path.exists() else None


def _sass_counts(compiled, tmp_path, ops):
    cubin = tmp_path / "kernel.cubin"
    cubin.write_bytes(compiled.__cubin__)
    text = subprocess.run(
        [str(_nvdisasm()), "-c", str(cubin)], capture_output=True, text=True, check=True
    ).stdout
    return {op: len(re.findall(re.escape(op), text)) for op in ops}


def _compile_variants(g, options):
    """Every GEMM variant the module serves, compiled for SM90: ``(name,
    transposes, compiled)``."""
    import cutlass

    row_scale = {False: torch.empty(2, 4), True: torch.empty(4)}
    postact = torch.empty(4, 64, dtype=torch.bfloat16)
    row_scaled = g.GemmSm90BlockFP8TRowScale
    variants = []
    for varlen in (False, True):
        keys = g._epilogue_keys(row_scaled, {"xs": row_scale[varlen]})
        for cluster_m in (1, 2):
            for transpose_b in (True, False):
                compiled = g.compile_gemm(
                    row_scaled,
                    keys,
                    transpose_b=transpose_b,
                    varlen=varlen,
                    cluster_m=cluster_m,
                    arch=SM90,
                    options=options,
                )
                name = f"row_scaled varlen={varlen} cluster={cluster_m}"
                variants.append((name, transpose_b, compiled))
    xs = row_scale[True]
    fused = [
        (row_scaled, {"xs": xs}, {"c_dtype": cutlass.BFloat16}, False),
        (
            g.epilogue_gemm_class(g.row_scaled_swiglu),
            {"xs": xs, "limit": 1.0, "postact": postact},
            {"concat": True},
            False,
        ),
        (
            g.epilogue_gemm_class(g.row_scaled_swiglu_add),
            {"xs": xs, "limit": 1.0, "postact": postact},
            {"concat": True, "c_dtype": cutlass.BFloat16},
            False,
        ),
        (
            g.epilogue_gemm_class(g.row_scaled_dswiglu_scored),
            {
                "xs": xs,
                "score": xs,
                "limit": 1.0,
                "postact": postact,
                "dscore": torch.empty(4, 2),
            },
            {"packed_cd": True, "c_dtype": cutlass.BFloat16},
            True,
        ),
    ]
    for gemm_cls, values, extra, transpose_b in fused:
        compiled = g.compile_gemm(
            gemm_cls,
            g._epilogue_keys(gemm_cls, values),
            transpose_b=transpose_b,
            varlen=True,
            arch=SM90,
            options=options,
            **extra,
        )
        variants.append((gemm_cls.__name__, transpose_b, compiled))
    return variants


def test_the_kernels_compile_for_sm90(tmp_path):
    """Every kernel traces and compiles for SM90 with no device: the transpose
    is 8 ldmatrix.x4.trans, 32 prmt and 8 stmatrix.x4 per 16 KB stage, each
    GEMM's K tile 4 FP8 wgmmas and 64 FFMA (the promotion), nothing spills."""
    g = _gemm_module()
    options = f"--keep-cubin --dump-dir {tmp_path}"
    ops = ("LDSM.16.MT88.4", "STSM.16.M88.4", "PRMT", "QGMMA", "FFMA", "STL", "LDL")
    transpose = g.compile_tile_transpose(SM90, options)
    variants = _compile_variants(g, options)
    if _nvdisasm() is None:
        pytest.skip("nvdisasm (nvidia-cuda-nvdisasm) is not installed")
    counts = _sass_counts(transpose, tmp_path, ops)
    assert counts["LDSM.16.MT88.4"] == 8 and counts["STSM.16.M88.4"] == 8
    assert counts["PRMT"] == 32
    assert counts["STL"] == counts["LDL"] == 0
    for name, transposes, compiled in variants:
        counts = _sass_counts(compiled, tmp_path, ops)
        # The transposing warpgroup's loop holds the one transpose.
        assert counts["LDSM.16.MT88.4"] == (8 if transposes else 0), name
        assert counts["QGMMA"] == 4 and counts["FFMA"] >= 64, name
        assert counts["STL"] == counts["LDL"] == 0, name


# ---- stages 1-3: the kernels, on SM90 -------------------------------------------


def _sm90() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return False
    try:
        import cutlass  # noqa: F401
        import quack  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


hopper = pytest.mark.skipif(
    not _sm90(), reason="needs an SM90 GPU with cutlass, quack and triton"
)


def _relative(actual, expected):
    actual, expected = actual.detach().double(), expected.detach().double()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-6))


def _worst_relative(actual, expected, row_bounds):
    """The largest relative error over the output's cells: the rows between
    consecutive ``row_bounds`` by each 128-column block."""
    actual = actual.reshape(-1, actual.shape[-1])
    expected = expected.reshape(-1, expected.shape[-1])
    worst = 0.0
    for lo, hi in itertools.pairwise(row_bounds):
        for col in range(0, expected.shape[1], 128):
            if hi > lo:
                cell = (slice(lo, hi), slice(col, col + 128))
                worst = max(worst, _relative(actual[cell], expected[cell]))
    return worst


def _source_index(width, order, device):
    index = torch.arange(width, device=device)
    return index - index % 16 + torch.tensor(order, device=device)[index % 16]


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("experts, k, n", [(1, 128, 128), (3, 272, 400)])
def test_the_tile_transpose_kernel(experts, k, n):
    """Every byte of every tile, partial tiles included (TMA zero-fills them
    and clips the store)."""
    g = _gemm_module()
    torch.manual_seed(0)
    raw = torch.randint(0, 256, (experts, k, n), dtype=torch.uint8, device="cuda")
    tiles = g.transposed_tiles(raw.view(torch.float8_e4m3fn))
    source_k = _source_index(k, g.K_ORDER, raw.device)
    source_n = _source_index(n, g.N_ORDER, raw.device)
    expected = raw[:, source_k][:, :, source_n].transpose(1, 2)
    assert tiles.shape == (experts, n, k)
    assert torch.equal(tiles.view(torch.uint8), expected)


def _block_weight(experts, k, n, exponents=(-3, 3)):
    """A block-scaled weight ``[E, K, N]``: Gaussian values times a power of
    two per 128x128 block, ``2**e`` for ``e`` drawn from ``exponents``, and
    one all-zero block when there are several. ``(data, scale, exact)``."""
    from mlops.kernels.block_fp8 import dequantize_blocks, quantize_blocks

    blocks = (experts, -(-k // 128), -(-n // 128))
    low, high = exponents
    factor = torch.randint(low, high + 1, blocks, device="cuda").float().exp2()
    if factor.numel() > 1:
        factor[-1, -1, 0] = 0.0
    factor = factor.repeat_interleave(128, 1).repeat_interleave(128, 2)[:, :k, :n]
    data, scale = quantize_blocks(torch.randn(experts, k, n, device="cuda") * factor)
    return data, scale, dequantize_blocks(data, scale)


def _ordered_rows(rows, *, gather=None, gate_up=False):
    """Quantized rows in the transposed GEMM's K order, with the logical
    (W-row-ordered) dequantized rows the reference multiplies."""
    quantize_rows = _quantizer()
    data, scales = quantize_rows.quantize_rows_fp8(
        rows, gather=gather, transposed_gemm=True, gate_up=gate_up
    )
    order = quantize_rows.source_columns(
        data.shape[1], transposed_gemm=True, device=data.device
    )
    logical = torch.empty_like(data.float())
    logical[:, order] = data.float()
    return data, scales, logical * scales[:, None]


def _dense_case(experts, m, k, n, cluster_m, exponents=(-3, 3)):
    g = _gemm_module()
    data, scale, exact = _block_weight(experts, k, n, exponents)
    rows = torch.randn(experts * m, k, device="cuda", dtype=torch.bfloat16)
    a, a_scale, logical = _ordered_rows(rows)
    out = g.block_fp8_transposed_gemm(
        a.view(experts, m, k),
        a_scale.view(experts, m),
        data,
        scale,
        cluster_m=cluster_m,
    )
    expected = torch.bmm(logical.view(experts, m, k).double(), exact.double())
    assert out.dtype is torch.bfloat16 and out.shape == (experts, m, n)
    return _worst_relative(out, expected, list(range(0, experts * m + 1, m)))


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize(
    "experts, m, k, n",
    [(1, 128, 128, 128), (2, 64, 256, 384), (3, 200, 384, 272), (2, 256, 4096, 256)],
)
@pytest.mark.parametrize("cluster_m", [1, 2])
def test_the_dense_gemm_matches_the_dequantized_product(experts, m, k, n, cluster_m):
    """Block scales a few powers of two apart, so every block's scale shows
    in every output cell; partial M, N and K tiles."""
    torch.manual_seed(1)
    assert _dense_case(experts, m, k, n, cluster_m) < 1e-2


@pytest.mark.gpu
@hopper
def test_the_promotion_spans_extreme_block_scales():
    """Block scales from 2**-14 to 2**14 (and a zero block) in one reduction."""
    torch.manual_seed(4)
    assert _dense_case(2, 256, 1024, 256, 1, exponents=(-14, 14)) < 1e-2


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("k, n", [(384, 256), (4096, 512)])
@pytest.mark.parametrize("gate_up", [False, True])
@pytest.mark.parametrize("cluster_m", [1, 2])
def test_the_varlen_gemm_matches_the_dequantized_product(k, n, gate_up, cluster_m):
    """Expert-sorted rows gathered from the tokens while quantized; an empty
    expert, partial row tiles, and the up projection's interleaved gate/up
    rows."""
    g = _gemm_module()
    torch.manual_seed(2)
    counts = [0, 70, 131, 15, 260]
    bounds = [0, *torch.tensor(counts).cumsum(0).tolist()]
    offsets = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    data, scale, exact = _block_weight(len(counts), k, n)
    tokens = torch.randn(300, k, device="cuda", dtype=torch.bfloat16)
    gather = torch.randint(0, 300, (bounds[-1],), device="cuda", dtype=torch.int32)
    a, a_scale, logical = _ordered_rows(tokens, gather=gather, gate_up=gate_up)
    out = g.block_fp8_transposed_gemm(
        a, a_scale, data, scale, offsets=offsets, cluster_m=cluster_m
    )
    expected = torch.zeros(bounds[-1], n, device="cuda", dtype=torch.float64)
    for expert in range(len(counts)):
        lo, hi = bounds[expert], bounds[expert + 1]
        expected[lo:hi] = logical[lo:hi].double() @ exact[expert].double()
    assert out.dtype is torch.bfloat16 and out.shape == (bounds[-1], n)
    assert _worst_relative(out, expected, bounds) < 1e-2


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize(
    "transposed_gemm, gate_up", [(True, False), (True, True), (False, True)]
)
@pytest.mark.parametrize("width", [256, 160])
def test_the_ordered_quantizer_kernel(transposed_gemm, gate_up, width):
    """The Triton kernel's ordered rows are its plain rows, reordered, and
    match the PyTorch reference: a row that fills its block (reordered in
    registers) and one that does not (read column by column)."""
    quantize_rows = _quantizer()
    torch.manual_seed(3)
    value = torch.randn(300, width, device="cuda", dtype=torch.bfloat16)
    gather = torch.randint(0, 300, (200,), device="cuda", dtype=torch.int32)
    data, scales = quantize_rows.quantize_rows_fp8(
        value, gather=gather, transposed_gemm=transposed_gemm, gate_up=gate_up
    )
    plain, plain_scales = quantize_rows.quantize_rows_fp8(value, gather=gather)
    columns = quantize_rows.source_columns(
        width, transposed_gemm=transposed_gemm, gate_up=gate_up, device="cuda"
    )
    assert torch.equal(scales, plain_scales)
    assert torch.equal(data.view(torch.uint8), plain[:, columns].view(torch.uint8))
    reference, reference_scales = quantize_rows.quantize_rows_fp8_reference(
        value, gather=gather, transposed_gemm=transposed_gemm, gate_up=gate_up
    )
    torch.testing.assert_close(scales, reference_scales)
    # At most one E4M3 step apart (rounding of equal FP32 values agrees).
    torch.testing.assert_close(
        data.float(), reference.float(), rtol=0.125, atol=2.0**-9
    )


# ---- the projections and the fused SwiGLU epilogues, on SM90 --------------------

COUNTS = [0, 70, 131, 15, 260]  # an empty expert, partial row tiles


def _routed_rows(k, *, seed, ordered=False):
    """Expert-sorted rows gathered from 300 tokens and quantized: ``(bounds,
    offsets, rows, row_scale, logical)``, ``logical`` the dequantized rows in
    natural column order."""
    torch.manual_seed(seed)
    bounds = [0, *torch.tensor(COUNTS).cumsum(0).tolist()]
    offsets = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    tokens = torch.randn(300, k, device="cuda", dtype=torch.bfloat16)
    gather = torch.randint(0, 300, (bounds[-1],), device="cuda", dtype=torch.int32)
    if ordered:
        rows, row_scale, logical = _ordered_rows(tokens, gather=gather)
    else:
        rows, row_scale = _quantizer().quantize_rows_fp8(tokens, gather=gather)
        logical = rows.float() * row_scale[:, None]
    return bounds, offsets, rows, row_scale, logical


def _per_expert(bounds, logical, exact, *, transposed):
    """float64 ``rows @ W[e]ᵀ`` (``W[e]`` when ``transposed``) per expert."""
    n = exact.shape[2] if transposed else exact.shape[1]
    out = torch.zeros(bounds[-1], n, device="cuda", dtype=torch.float64)
    for expert in range(len(bounds) - 1):
        lo, hi = bounds[expert], bounds[expert + 1]
        weight = exact[expert].double()
        out[lo:hi] = logical[lo:hi].double() @ (weight if transposed else weight.T)
    return out


def _swiglu_reference(gate, up, limit):
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return torch.nn.functional.silu(gate) * up


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("k, n", [(384, 256), (4096, 272)])
@pytest.mark.parametrize("addend", [False, True])
@pytest.mark.parametrize("cluster_m", [1, 2])
def test_the_projection_matches_the_dequantized_product(k, n, addend, cluster_m):
    """``W [E, N, K]`` K-major, read without a transpose; the LoRA addend."""
    g = _gemm_module()
    bounds, offsets, rows, row_scale, logical = _routed_rows(k, seed=5)
    data, scale, exact = _block_weight(len(COUNTS), n, k)
    c = None
    if addend:
        c = torch.randn(bounds[-1], n, device="cuda", dtype=torch.bfloat16)
    out = g.block_fp8_gemm(
        rows, row_scale, data, scale, offsets=offsets, c=c, cluster_m=cluster_m
    )
    expected = _per_expert(bounds, logical, exact, transposed=False)
    if addend:
        expected += c.double()
    assert out.dtype is torch.bfloat16 and out.shape == (bounds[-1], n)
    assert _worst_relative(out, expected, bounds) < 1e-2


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("cluster_m", [1, 2])
def test_the_dense_projection_matches_the_dequantized_product(cluster_m):
    """Equal-row experts as a batch, partial row and column tiles."""
    g = _gemm_module()
    torch.manual_seed(8)
    experts, m, k, n = 3, 200, 384, 272
    data, scale, exact = _block_weight(experts, n, k)
    rows = torch.randn(experts * m, k, device="cuda", dtype=torch.bfloat16)
    a, a_scale = _quantizer().quantize_rows_fp8(rows)
    logical = (a.float() * a_scale[:, None]).view(experts, m, k)
    out = g.block_fp8_gemm(
        a.view(experts, m, k),
        a_scale.view(experts, m),
        data,
        scale,
        cluster_m=cluster_m,
    )
    expected = torch.bmm(logical.double(), exact.double().transpose(1, 2))
    assert out.dtype is torch.bfloat16 and out.shape == (experts, m, n)
    assert _worst_relative(out, expected, list(range(0, experts * m + 1, m))) < 1e-2


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("i", [128, 192])
@pytest.mark.parametrize("limit", [None, 1.5])
@pytest.mark.parametrize("addend", [False, True])
def test_the_up_projection_interleaves_and_applies_swiglu(i, limit, addend):
    """``w13``'s gate rows then up rows, the output interleaved; each N tile
    folds its gate and up columns at their own block scales (I = 192 puts the
    up rows of a tile in a different block row from the gate rows')."""
    g = _gemm_module()
    k = 256
    bounds, offsets, rows, row_scale, logical = _routed_rows(k, seed=6)
    data, scale, exact = _block_weight(len(COUNTS), 2 * i, k)
    c = None
    if addend:
        c = torch.randn(bounds[-1], 2 * i, device="cuda", dtype=torch.bfloat16)
    h, act = g.block_fp8_gemm_swiglu(
        rows, row_scale, data, scale, offsets=offsets, c=c, limit=limit
    )
    plain = _per_expert(bounds, logical, exact, transposed=False)
    expected = torch.empty_like(plain)
    expected[:, 0::2], expected[:, 1::2] = plain[:, :i], plain[:, i:]
    if addend:
        expected += c.double()
    assert h.shape == (bounds[-1], 2 * i) and act.shape == (bounds[-1], i)
    assert _worst_relative(h, expected, bounds) < 1e-2
    reference = _swiglu_reference(expected[:, 0::2], expected[:, 1::2], limit)
    assert _worst_relative(act, reference, bounds) < 2e-2


def _dswiglu_reference(r, h, score, limit):
    """``(dh, act * score, <act, r>)`` of the clamped SwiGLU at ``h``."""
    gate, up = h[:, 0::2].double(), h[:, 1::2].double()
    clamped_gate, clamped_up = gate, up
    if limit is not None:
        clamped_gate, clamped_up = gate.clamp(max=limit), up.clamp(-limit, limit)
    sigmoid = torch.sigmoid(clamped_gate)
    silu = clamped_gate * sigmoid
    act = silu * clamped_up
    dout = r * score[:, None]
    dgate = dout * clamped_up * sigmoid * (1 + clamped_gate * (1 - sigmoid))
    dup = dout * silu
    if limit is not None:
        dgate = torch.where(gate <= limit, dgate, 0.0)
        dup = torch.where((up <= limit) & (up >= -limit), dup, 0.0)
    dh = torch.empty(h.shape, device=h.device, dtype=torch.float64)
    dh[:, 0::2], dh[:, 1::2] = dgate, dup
    return dh, act * score[:, None], (act * r).sum(dim=1)


@pytest.mark.gpu
@hopper
@pytest.mark.parametrize("limit", [None, 1.0])
def test_the_down_input_gradient_fuses_the_swiglu_backward(limit):
    """``r = dy @ w2`` through the transposing GEMM, then the SwiGLU backward
    against the saved interleaved ``h`` in the epilogue."""
    g = _gemm_module()
    width, i = 384, 192
    bounds, offsets, rows, row_scale, logical = _routed_rows(
        width, seed=7, ordered=True
    )
    data, scale, exact = _block_weight(len(COUNTS), width, i)
    h = torch.randn(bounds[-1], 2 * i, device="cuda", dtype=torch.bfloat16) * 2
    score = torch.rand(bounds[-1], device="cuda", dtype=torch.float32)
    dh, act, dscore = g.block_fp8_transposed_gemm_dswiglu(
        rows, row_scale, data, scale, h, score, offsets=offsets, limit=limit
    )
    r = _per_expert(bounds, logical, exact, transposed=True)
    dh_ref, act_ref, dscore_ref = _dswiglu_reference(r, h, score.double(), limit)
    assert dh.dtype is torch.bfloat16 and dh.shape == h.shape
    assert act.shape == (bounds[-1], i) and dscore.shape == (bounds[-1],)
    assert _worst_relative(dh, dh_ref, bounds) < 2e-2
    assert _worst_relative(act, act_ref, bounds) < 1e-2
    assert _relative(dscore, dscore_ref) < 1e-2
