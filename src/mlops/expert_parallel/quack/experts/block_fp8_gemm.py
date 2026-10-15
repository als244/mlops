"""Grouped GEMMs on block-scaled E4M3 expert weights, in either orientation.

A `BlockFP8Weight` stores ``W [E, out, in]`` as E4M3 with one FP32 scale per
128x128 block (``w_scale [E, ceil(out/128), ceil(in/128)]``). Per expert
``e`` and with E4M3 rows ``a`` scaled by one FP32 ``a_scale`` per row:

* a projection, ``(a * a_scale) @ W[e]ᵀ`` (`block_fp8_gemm`; the up
  projection with gate and up interleaved and the SwiGLU in the epilogue,
  `block_fp8_gemm_swiglu`), reduces over ``in``, W's contiguous axis;
* an input gradient, ``(a * a_scale) @ W[e]`` (`block_fp8_transposed_gemm`;
  with the SwiGLU backward in the epilogue,
  `block_fp8_transposed_gemm_dswiglu`), reduces over ``out``, which is not.

Both are `GemmSm90BlockFP8T`, Quack's SM90 GEMM with its own producers and
mainloop: 128x128x128 tiles, each K tile one scale block, its four wgmmas
accumulated from zero and folded into the running FP32 sum at the block's
scale; the row scales and the rest apply in the epilogue (Quack's epilogue
machinery, including functions minted by its ``gemm_epilogue`` frontend).

Hopper's FP8 wgmma reads only K-major shared-memory operands, so for an input
gradient a transposing warpgroup rewrites each weight tile in shared memory, in
place, after TMA loads it as stored, with FlashAttention-3's FP8 V transpose:
``ldmatrix.x4.trans.b16``, two ``prmt`` per register pair and
``stmatrix.x4.b16``, then an arrive on the stage's "transposed" barrier the
MMA warpgroups wait on. That sequence is free of shared-memory bank conflicts
on both sides and leaves each 16-wide group of K and of N permuted:

* K: stored position ``p`` holds row ``TRANSPOSED_GEMM_ORDER[p]`` of the
  group. The row operand must carry the same order:
  ``quantize_rows_fp8(..., transposed_gemm=True)`` writes it.
* N: accumulator column ``p`` holds column ``N_ORDER[p]`` of the group (bits 0
  and 3 of ``p`` swapped). One in-register swap per column pair restores the
  standard order before the epilogue.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import cutlass
import cutlass.utils.hopper_helpers as sm90_utils_og
import torch
from cutlass import Boolean, Float32, Int32, const_expr, cute, pipeline
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.experimental import primitives as prims
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum, SmemPartition
from quack import copy_utils, layout_utils
from quack import sm90_utils as quack_sm90_utils
from quack.compile_utils import fake_batched, make_fake_stream, make_fake_tensor
from quack.cute_dsl_utils import (
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.epilogue.frontend import _pinned_visit_kind, gemm_epilogue
from quack.epilogue.math import pack, unpack
from quack.epilogue.mixin import ComposableEpiMixin
from quack.epilogue.ops import ColVecLoad, ColVecReduce, Scalar
from quack.epilogue.visit import _EpiModMixinBase
from quack.gemm_runtime.host import FakeArgCtx
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import (
    make_fake_gemm_tensors,
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
)
from quack.pipeline import make_pipeline_state
from quack.rounding import RoundingMode
from quack.varlen_utils import VarlenArguments, VarlenManager

from mlops.expert_parallel.quack.experts.swiglu_limit import (
    dswiglu_clamped,
    swiglu_clamped,
)
from mlops.expert_parallel.quack.kernels.quantize_rows import TRANSPOSED_GEMM_ORDER

TILE_M, TILE_N, TILE_K = 128, 128, 128
BLOCK = 128  # the weight's scale block, along both axes

# Within every aligned group of 16 rows of the transposed weight tile (and 16
# columns of the accumulator), position p holds column N_ORDER[p].
N_ORDER = tuple(8 * (p & 1) + (p & 6) + (p >> 3) for p in range(16))
K_ORDER = TRANSPOSED_GEMM_ORDER

# FlashAttention-3's transpose copies (hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp,
# the head-dim-multiple-of-64 case): (shape, stride) of the thread and value
# layouts, and the tile each copy partitions.
LDSM_THREADS = ((32, 4, 1, 1), (4, 1, 0, 0))
LDSM_VALUES = ((2, 2, 1, 4), (1, 2, 16, 4))
LDSM_TILE = (64, 8)
STSM_THREADS = ((8, 4, 4, 1), (4, 1, 32, 0))
STSM_VALUES = ((1, 4, 2, 2), (0, 1, 4, 8))
STSM_TILE = (8, 16)
# Byte selectors of the two prmt per register pair (x, y): {x0, x2, y0, y2}
# and {x1, x3, y1, y3}.
PRMT_SELECTORS = (0x6420, 0x7531)
# The transposing warpgroup's barrier between reading and rewriting a stage (in
# place). Quack's GEMM barriers use ids 1-8.
TRANSPOSE_BARRIER_ID = 9
TRANSPOSE_THREADS = 128


def _layout(shape_stride):
    shape, stride = shape_stride
    return cute.make_layout(shape, stride=stride)


def transpose_copies(dtype):
    """The tiled copies of the in-place tile transpose: ``ldmatrix.x4.trans``
    (shared memory to registers) and ``stmatrix.x4`` (back), for 128 threads.
    Call inside a CuTe-DSL trace."""
    s2r = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), dtype
        ),
        _layout(LDSM_THREADS),
        _layout(LDSM_VALUES),
    )
    r2s = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), dtype
        ),
        _layout(STSM_THREADS),
        _layout(STSM_VALUES),
    )
    return s2r, r2s


def mn_major_view(tile):
    """``(N, K, ...)`` MN-major tile → the domain the ldmatrix copy partitions."""
    return cute.flat_divide(tile, LDSM_TILE)


def k_major_view(tile):
    """``(N, K, ...)`` K-major tile → the domain the stmatrix copy partitions.

    The N mode goes through ``N_ORDER``'s permutation: logical row
    ``t = b + 4a + 8h`` of each 16 (``b < 4``) lands on row ``2b + a + 8h``, so
    every stmatrix writes 8 rows of distinct swizzle phase and the accumulator
    columns need only in-register swaps afterwards."""
    width = cute.size(tile, mode=[0])
    permuted = cute.make_layout(((4, 2, 2), width // 16), stride=((2, 1, 8), 16))
    tiler = (permuted,) + (None,) * (cute.rank(tile) - 1)
    return cute.flat_divide(cute.composition(tile, tiler), STSM_TILE)


def make_tile_transposer(sB_mn, sB_k, tidx):
    """``transpose(stage)``: rewrite stage ``stage`` of the staged weight tile
    ``sB_mn`` (``(N, K, STAGE)``, MN-major, as TMA wrote it) in place as its
    K-major view ``sB_k`` over the same bytes. All 128 threads of the calling
    warpgroup (``tidx`` in ``[0, 128)``) take part; the stage's TMA bytes must
    have landed. Ends with the proxy fence that makes the writes visible to
    wgmma; the caller signals the consumers."""
    s2r, r2s = transpose_copies(sB_mn.element_type)
    # Plain partitions of the swizzled tensors (the swizzle applies to each
    # full address): the same partitions the CPU tests read the maps from.
    src = s2r.get_slice(tidx).partition_S(mn_major_view(sB_mn))
    dst = r2s.get_slice(tidx).partition_D(k_major_view(sB_k))
    # ((16, 1), steps, STAGE): one 16-byte ldmatrix / stmatrix per step.
    src = cute.group_modes(src, 1, cute.rank(src) - 1)
    dst = cute.group_modes(dst, 1, cute.rank(dst) - 1)
    steps = cute.size(dst, mode=[1])
    assert cute.size(src, mode=[1]) == steps
    frag = cute.make_rmem_tensor(dst[None, None, 0].shape, sB_mn.element_type)
    words = cute.recast_tensor(frag, Int32)
    barrier = pipeline.NamedBarrier(
        barrier_id=TRANSPOSE_BARRIER_ID, num_threads=TRANSPOSE_THREADS
    )

    # Plain Python loops: this helper is not preprocessed, they unroll at trace.
    def transpose(stage):
        for i in range(steps):
            cute.copy(s2r, src[None, i, stage], frag[None, i])
        # In place: every read of the stage before any write.
        barrier.arrive_and_wait()
        for w in range(cute.size(words) // 2):
            x, y = words[2 * w], words[2 * w + 1]
            words[2 * w] = prims.prmt(
                x, PRMT_SELECTORS[0], prims.PermuteMode.DEFAULT, hi=y
            )
            words[2 * w + 1] = prims.prmt(
                x, PRMT_SELECTORS[1], prims.PermuteMode.DEFAULT, hi=y
            )
        for i in range(steps):
            cute.copy(r2s, frag[None, i], dst[None, i, stage])
        cute.arch.fence_view_async_shared()

    return transpose


def _weight_smem_layouts(dtype, stages):
    """The staged ``(TILE_N, TILE_K, stages)`` weight tile as TMA writes it
    (MN-major) and as wgmma reads it (K-major): same bytes, same stage stride."""
    mn_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_og.get_smem_layout_atom(LayoutEnum.COL_MAJOR, dtype, TILE_N), dtype
    )
    k_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_og.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, dtype, TILE_K), dtype
    )
    shape = (TILE_N, TILE_K, stages)
    return (
        cute.tile_to_shape(mn_atom, shape, order=(1, 0, 2)),
        cute.tile_to_shape(k_atom, shape, order=(0, 1, 2)),
    )


# ---- the transpose alone (one tile per CTA) ------------------------------------


class TileTransposeKernel:
    """One CTA per ``(TILE_N, TILE_K)`` weight tile: TMA load, the in-place
    transpose, TMA store of the tile as it then sits in shared memory. Tests
    the transposer by itself; the GEMM runs the same `make_tile_transposer`."""

    @cute.jit
    def __call__(self, mW, mOut, stream):
        # mW (l, k, n) as stored, n contiguous; mOut (l, n, k), k contiguous.
        mW_nkl = layout_utils.select(mW, [2, 1, 0])
        mOut_nkl = layout_utils.select(mOut, [1, 2, 0])
        dtype = mW.element_type
        mn_layout, k_layout = _weight_smem_layouts(dtype, 1)
        load_atom, load_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mW_nkl,
            cute.slice_(mn_layout, (None, None, 0)),
            (TILE_N, TILE_K),
        )
        store_atom, store_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mOut_nkl,
            cute.slice_(k_layout, (None, None, 0)),
            (TILE_N, TILE_K),
        )
        self.tile_bytes = cute.size_in_bytes(
            dtype, cute.slice_(mn_layout, (None, None, 0))
        )

        @cute.struct
        class SharedStorage:
            barrier: cute.struct.MemRange[cutlass.Int64, 1]
            sB: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(mn_layout)], 1024
            ]

        self.shared_storage = SharedStorage
        grid = (
            cute.ceil_div(cute.size(mW_nkl, mode=[0]), TILE_N),
            cute.ceil_div(cute.size(mW_nkl, mode=[1]), TILE_K),
            cute.size(mW_nkl, mode=[2]),
        )
        self.kernel(
            load_atom, load_tensor, store_atom, store_tensor, mn_layout, k_layout
        ).launch(grid=grid, block=[TRANSPOSE_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        load_atom: cute.CopyAtom,
        mW_nkl: cute.Tensor,
        store_atom: cute.CopyAtom,
        mOut_nkl: cute.Tensor,
        mn_layout: cute.ComposedLayout,
        k_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        n_block, k_block, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        barrier = storage.barrier.data_ptr()
        sB_mn = storage.sB.get_tensor(mn_layout.outer, swizzle=mn_layout.inner)
        sB_k = storage.sB.get_tensor(k_layout.outer, swizzle=k_layout.inner)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        gW = cute.local_tile(
            mW_nkl[None, None, batch], (TILE_N, TILE_K), (n_block, k_block)
        )
        gOut = cute.local_tile(
            mOut_nkl[None, None, batch], (TILE_N, TILE_K), (n_block, k_block)
        )
        load, _, _ = copy_utils.tma_get_copy_fn(
            load_atom,
            0,
            cute.make_layout(1),
            gW,
            sB_mn[None, None, 0],
            single_stage=True,
        )
        store, _, _ = copy_utils.tma_get_copy_fn(
            store_atom,
            0,
            cute.make_layout(1),
            sB_k[None, None, 0],
            gOut,
            single_stage=True,
        )
        transpose = make_tile_transposer(sB_mn, sB_k, tidx)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.tile_bytes)
            load(tma_bar_ptr=barrier)
        cute.arch.mbarrier_wait(barrier, 0)
        transpose(0)
        cute.arch.sync_threads()
        if warp_idx == 0:
            store()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)


def _options(arch, extra):
    options = "--enable-tvm-ffi" + (f" --gpu-arch {arch}" if arch else "")
    return options + (f" {extra}" if extra else "")


@functools.cache
def compile_tile_transpose(arch=None, options=None):
    """The compiled `TileTransposeKernel`. ``arch`` (e.g. ``"sm_90a"``) pins
    the target, else the device's is used; ``options`` adds compile options
    (e.g. ``"--keep-cubin --dump-dir DIR"`` to inspect the code)."""
    l, k, n = (
        cute.sym_int(),
        cute.sym_int(divisibility=16),
        cute.sym_int(divisibility=16),
    )
    fp8 = cutlass.Float8E4M3FN
    weight = fake_batched(fp8, k, n, l, 1, 16)
    out = fake_batched(fp8, n, k, l, 1, 16)
    return cute.compile(
        TileTransposeKernel(),
        weight,
        out,
        make_fake_stream(),
        options=_options(arch, options),
    )


def transposed_tiles(weight):
    """The K-major tiles the in-kernel transpose leaves in shared memory, laid
    side by side: ``[E, N, K]`` E4M3 for an E4M3 ``weight [E, K, N]`` (N
    contiguous; K and N multiples of 16). Within every 16x16 block, element
    ``[e, n, k]`` holds ``weight[e, k0 + K_ORDER[k % 16], n0 + N_ORDER[n % 16]]``
    (``k0``, ``n0`` the block's corner)."""
    if weight.ndim != 3 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected an E4M3 weight [E, K, N]")
    experts, k, n = weight.shape
    if k % 16 or n % 16 or weight.stride(2) != 1:
        raise ValueError("K and N must be multiples of 16, N contiguous")
    out = torch.empty(experts, n, k, dtype=weight.dtype, device=weight.device)
    if out.numel():
        compile_tile_transpose()(weight, out)
    return out


# ---- the GEMM ------------------------------------------------------------------


class GemmSm90BlockFP8T(GemmSm90):
    """Quack's SM90 GEMM for a block-scaled E4M3 B.

    Cooperative, ``(TILE_M, TILE_N, TILE_K)`` tiles, two MMA warpgroups and a
    producer warpgroup whose first warp issues the TMA loads and runs the tile
    scheduler as in `GemmSm90`. The MMA warpgroups accumulate each K tile from
    zero and fold it into the running sum times the tile's block scale
    (``mSFB`` FP32, read per K tile); `GemmSm90`'s epilogue (mixins, C loads,
    TMA store) then runs on the sum.

    ``transpose_b`` (the input gradient): ``mB`` is the stored weight
    ``(N, K)`` N-major. A fourth warpgroup transposes each B stage in place
    once it lands (`make_tile_transposer`) and arrives on the stage's
    "transposed" barrier, which the MMA warpgroups wait on, while the TMA
    warp loads ahead by the whole pipeline; ``mA`` carries its columns in
    `K_ORDER`, and the accumulator's columns are put back in order before the
    epilogue. ``mSFB [L, ceil(K/128), ceil(N/128)]``.

    Otherwise (a projection): ``mB`` is K-major as the wgmma reads it,
    ``mSFB [L, ceil(N/128), ceil(K/128)]``. With ``concat_layout=("B",)`` B's
    rows are a gate half then an up half read interleaved (output column
    ``2j`` from gate row ``j``, ``2j + 1`` from up row ``j``): each N tile
    folds its even columns at the gate rows' block scale and its odd columns
    at the up rows'.

    Dense batched or varlen over M (expert offsets). No gather, split-K or
    pingpong.
    """

    def __init__(
        self,
        acc_dtype,
        a_dtype,
        tile_shape_mnk,
        cluster_shape_mnk,
        pingpong=False,
        is_persistent=True,
        gather_A=False,
        concat_layout=None,
        use_pdl=False,
        transpose_b=True,
        **kwargs,
    ):
        if pingpong or not is_persistent or gather_A or kwargs:
            raise ValueError(
                "GemmSm90BlockFP8T is cooperative and persistent, without gather"
            )
        if tuple(tile_shape_mnk) not in ((TILE_M, TILE_N), (TILE_M, TILE_N, TILE_K)):
            raise ValueError(f"tile shape must be ({TILE_M}, {TILE_N}, {TILE_K})")
        if cluster_shape_mnk[1:] != (1, 1) or cluster_shape_mnk[0] not in (1, 2):
            raise ValueError("cluster shape must be (1, 1, 1) or (2, 1, 1)")
        concat_layout = tuple(concat_layout or ())
        if concat_layout not in ((), ("B",)) or (concat_layout and transpose_b):
            raise ValueError("concat_layout is ('B',), on a K-major weight only")
        super().__init__(
            acc_dtype,
            a_dtype,
            (TILE_M, TILE_N, TILE_K),
            cluster_shape_mnk,
            pingpong=False,
            is_persistent=True,
            # The K-tile promotion with the block scales is this class's own.
            fp8_fast_accum=True,
            gather_A=False,
            concat_layout=concat_layout,
            use_pdl=use_pdl,
        )
        self.transpose_b = transpose_b
        if transpose_b:
            # Two producer warpgroups: the transposer (warps 8-11) and, as in
            # GemmSm90, the TMA loads and the tile scheduler (warp 12), so the
            # loads run ahead of the transposes by the whole pipeline.
            self.transpose_warp_id = self.mma_warp_groups * 4
            self.ab_load_warp_id = self.transpose_warp_id + 4
            self.threads_per_cta = (self.mma_warp_groups + 2) * 128
            # The scheduler's consumers beyond the MMA warps: the TMA warp
            # and the four transposing warps.
            self.num_ab_load_warps = 5
            # 128 * (load + transpose) + 256 * mma within 512 threads' 65536.
            self.num_regs_load, self.num_regs_transpose = 24, 56
            self.num_regs_mma = 216

    def _setup_tiled_mma(self):
        # Both operands K-major: B as given, or its transposed copy in smem.
        self.tiled_mma = sm90_utils_og.make_trivial_tiled_mma(
            self.mma_a_dtype,
            self.b_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, TILE_N),
            a_source=warpgroup.OperandSource.SMEM,
        )
        self.cta_tile_shape_mnk = (TILE_M, TILE_N, TILE_K)

    def _setup_attributes(self, epilogue_args):
        super()._setup_attributes(epilogue_args)
        if self.transpose_b:
            # The base built B's staged layout from its (MN-major) tensor: the
            # TMA view. The wgmma descriptors read the K-major view of the
            # same bytes.
            self.b_tma_smem_layout_staged, self.b_smem_layout_staged = (
                _weight_smem_layouts(self.b_dtype, self.ab_stage)
            )
        else:
            self.b_tma_smem_layout_staged = self.b_smem_layout_staged

    def make_bt_pipeline(self):
        """Full-only: the transposing threads arrive on a stage once it is
        rewritten; reuse is gated by the AB pipeline's empty barrier."""
        return pipeline.PipelineAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, TRANSPOSE_THREADS
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.mma_warp_groups * self.num_threads_per_warp_group,
            ),
            defer_sync=True,
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mC: cute.Tensor | None,
        epilogue_args: tuple,
        scheduler_args,
        varlen_args: VarlenArguments | None,
        stream,
        mSFA: cute.Tensor | None = None,
        mSFB: cute.Tensor | None = None,
    ):
        """``mSFB``: B's block scales, FP32 (see the class)."""
        assert mSFA is None and mSFB is not None
        mA, mB, mD, mC, epilogue_args = self.rotate_batch_last(
            mA, mB, mD, mC, epilogue_args, append_batch_if_2d=varlen_args is None
        )
        if const_expr("B" in self.concat_layout):
            mB = layout_utils.concat_to_interleave(mB, 1 - mB.leading_dim)
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type
        self.c_dtype = mC.element_type if mC is not None else None
        self.sf_dtype = None
        self.a_smem_dtype = self.a_dtype
        self.b_smem_dtype = self.b_dtype
        self.a_layout = LayoutEnum.from_tensor(mA)
        self.b_layout = LayoutEnum.from_tensor(mB)
        self.d_layout = LayoutEnum.from_tensor(mD)
        self.c_layout = LayoutEnum.from_tensor(mC) if mC is not None else None
        if const_expr(
            self.a_dtype != cutlass.Float8E4M3FN or self.b_dtype != self.a_dtype
        ):
            raise TypeError("A and B must be Float8E4M3FN")
        if const_expr(
            self.a_layout.sm90_mma_major_mode() != cute.nvgpu.OperandMajorMode.K
        ):
            raise TypeError("A must be K-major")
        b_major = (
            cute.nvgpu.OperandMajorMode.MN
            if const_expr(self.transpose_b)
            else cute.nvgpu.OperandMajorMode.K
        )
        if const_expr(self.b_layout.sm90_mma_major_mode() != b_major):
            raise TypeError("B must be N-major with transpose_b, else K-major")
        if const_expr(varlen_args is None):
            varlen_args = VarlenArguments()
        assert varlen_args.mAIdx is None and varlen_args.mCuSeqlensK is None
        varlen_m = varlen_args.mCuSeqlensM is not None
        self.varlen_m = varlen_m
        self.varlen_k = False

        self._setup_attributes(epilogue_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_tma_smem_layout = cute.slice_(self.b_tma_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = (
            self.make_tma_load_atoms_and_tensors(
                mA,
                mB,
                a_smem_layout,
                b_tma_smem_layout,
                False,
                varlen_m_zero_fill=varlen_m
                and self.epilogue_zero_fill_varlen_m(epilogue_args),
            )
        )
        self.num_tma_load_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_tma_smem_layout)
        tma_atom_d, tma_tensor_d, tma_atom_c, tma_tensor_c = (
            self.make_tma_epilogue_atoms_and_tensors(mD, mC, epilogue_args, varlen_m)
        )
        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)
        self.epi_load_bytes_per_stage = self.epi_smem_bytes(
            epilogue_args,
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.epi_smem_warp_shape_mnk(),
        ).c_stage
        if const_expr(mC is not None):
            c_smem_layout = cute.slice_(self.epi_c_smem_layout_staged, (None, None, 0))
            self.epi_load_bytes_per_stage += cute.size_in_bytes(
                self.c_dtype, c_smem_layout
            )

        TileSchedulerCls = self.get_scheduler_class(varlen_m=varlen_m)
        tile_sched_args = self.get_scheduler_arguments(
            mA, mB, mD, scheduler_args, varlen_args, epilogue_args
        )
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(
            tile_sched_params, scheduler_args.max_active_clusters
        )
        epi_smem_size = cute.cosize(self.epi_smem_layout_staged)
        epi_c_smem_size = (
            cute.cosize(self.epi_c_smem_layout_staged) if mC is not None else 0
        )

        @cute.struct
        class SharedStorage:
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype if self.c_dtype is not None else Int32,
                    epi_c_smem_size,
                ],
                self.buffer_align_bytes,
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_smem_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_smem_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mSFB,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            epilogue_params,
            varlen_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.b_tma_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
            tile_sched_params,
            TileSchedulerCls,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFB: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        tma_atom_c: cute.CopyAtom | None,
        mC_mnl: cute.Tensor | None,
        epilogue_params,
        varlen_params: VarlenManager.Params,
        cluster_layout_mnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        b_tma_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_c_smem_layout: cute.ComposedLayout | None,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        has_C = const_expr(mC_mnl is not None)
        has_epi_load = const_expr(self.epi_c_stage > 0)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
        )
        bt_pipeline = None
        if const_expr(self.transpose_b):
            bt_pipeline = self.make_bt_pipeline()
        epi_pipeline = None
        if const_expr(has_epi_load):
            epi_pipeline = self.make_epi_pipeline(
                tx_count=self.epi_load_bytes_per_stage
            )
        sched_pipeline = self.make_sched_pipeline(cluster_layout_mnk, varlen_k=False)
        sched_data = smem.allocate_tensor(
            Int32,
            cute.make_layout((4, self.sched_stage)),
            byte_alignment=16,
            partition=SmemPartition.RESERVED,
        )

        pipeline_init_arrive(
            cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True
        )

        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        # With transpose_b, one buffer and two views: TMA writes the MN-major
        # one, the transposer rewrites it in place as the K-major one the
        # wgmma descriptors read.
        sB_tma = sB
        if const_expr(self.transpose_b):
            sB_tma = storage.sB.get_tensor(
                b_tma_smem_layout.outer, swizzle=b_tma_smem_layout.inner
            )
        sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(
                epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner
            )
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        varlen_manager = VarlenManager.create(
            varlen_params,
            len_m_static=Int32(cute.size(mA_mkl, mode=[0])),
            len_k_static=Int32(cute.size(mB_nkl, mode=[1])),
            len_n_static=Int32(cute.size(mB_nkl, mode=[0])),
        )
        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if warp_idx == self.ab_load_warp_id:
                self._load(
                    ab_pipeline,
                    tma_atom_a,
                    mA_mkl,
                    tma_atom_b,
                    mB_nkl,
                    sA,
                    sB_tma,
                    varlen_manager,
                    TileSchedulerCls,
                )
        if const_expr(self.transpose_b):
            self._transpose_stages(
                ab_pipeline,
                bt_pipeline,
                sB,
                sB_tma,
                varlen_manager,
                TileSchedulerCls,
                warp_idx,
            )

        if warp_idx < self.mma_warp_groups * 4:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            self._consume(
                ab_pipeline,
                bt_pipeline,
                epi_pipeline,
                tiled_mma,
                mSFB,
                tma_atom_d,
                mD_mnl,
                tma_atom_c,
                mC_mnl,
                epilogue_params,
                sA,
                sB,
                sD,
                sC,
                epi_smem_tensors,
                varlen_manager,
                TileSchedulerCls,
                warp_idx,
            )

    @cute.jit
    def _load(
        self,
        ab_pipeline,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        sA: cute.Tensor,
        sB_tma: cute.Tensor,
        varlen_manager: VarlenManager,
        TileSchedulerCls: Callable,
    ):
        """The TMA warp: A and B tiles into the AB stages, and the tile
        scheduler, as in `GemmSm90`."""
        if const_expr(self.use_pdl):
            prims.griddepcontrol(prims.GridDepAction.WAIT)
        is_scheduler_warp = True
        if const_expr(cute.size(self.cluster_shape_mnk) > 1):
            is_scheduler_warp = cute.arch.block_idx_in_cluster() == 0
        a_tma_multicast = {
            "cluster_shape": self.cluster_shape_mnk[:2],
            "multicast_dim": "M",
        }
        b_tma_multicast = {
            "cluster_shape": self.cluster_shape_mnk[:2],
            "multicast_dim": "N",
        }
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        load_state = make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        while work_tile.is_valid_tile:
            tile_coord_mnkl = work_tile.tile_idx
            batch_idx = tile_coord_mnkl[3]
            gA_mk = cute.local_tile(
                varlen_manager.offset_batch_A(mA_mkl, batch_idx),
                cute.select(self.cta_tile_shape_mnk, [0, 2]),
                (tile_coord_mnkl[0], None),
            )
            copy_A = copy_utils.tma_get_block_copy_fn(
                tma_atom_a,
                src_tensor=gA_mk,
                dst_tensor=sA,
                tma_multicast=a_tma_multicast,
            )
            gB_nk = cute.local_tile(
                varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                cute.select(self.cta_tile_shape_mnk, [1, 2]),
                (tile_coord_mnkl[1], None),
            )
            copy_B = copy_utils.tma_get_block_copy_fn(
                tma_atom_b,
                src_tensor=gB_nk,
                dst_tensor=sB_tma,
                tma_multicast=b_tma_multicast,
            )
            k_tile_cnt = cute.ceil_div(
                varlen_manager.len_k(batch_idx), self.cta_tile_shape_mnk[2]
            )
            for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                ab_pipeline.producer_acquire(load_state)
                tma_bar_ptr = ab_pipeline.producer_get_barrier(load_state)
                copy_A(k_tile, load_state.index, tma_bar_ptr=tma_bar_ptr)
                copy_B(k_tile, load_state.index, tma_bar_ptr=tma_bar_ptr)
                ab_pipeline.producer_commit(load_state)
                load_state.advance()
            tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
            work_tile = tile_scheduler.get_current_work()
        ab_pipeline.producer_tail(load_state)
        if is_scheduler_warp:
            tile_scheduler.producer_tail()

    @cute.jit
    def _transpose_stages(
        self,
        ab_pipeline,
        bt_pipeline,
        sB: cute.Tensor,
        sB_tma: cute.Tensor,
        varlen_manager: VarlenManager,
        TileSchedulerCls: Callable,
        warp_idx: Int32,
    ):
        """The transposing warpgroup (the calling warps among it): each B stage
        in load order, once its TMA landed, then the stage's "transposed"
        arrive for the MMA warpgroups. Reuse of a stage is gated by the AB
        pipeline: the TMA warp refills it only after the MMA warpgroups,
        which wait on this arrive, release it."""
        if warp_idx >= self.transpose_warp_id and warp_idx < self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_transpose)
            self._transpose_loop(
                ab_pipeline, bt_pipeline, sB, sB_tma, varlen_manager, TileSchedulerCls
            )

    @cute.jit
    def _transpose_loop(
        self,
        ab_pipeline,
        bt_pipeline,
        sB: cute.Tensor,
        sB_tma: cute.Tensor,
        varlen_manager: VarlenManager,
        TileSchedulerCls: Callable,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        transpose = make_tile_transposer(
            sB_tma, sB, tidx - self.transpose_warp_id * cute.arch.WARP_SIZE
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        landed_state = make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        transposed_state = make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        while work_tile.is_valid_tile:
            batch_idx = work_tile.tile_idx[3]
            k_tile_cnt = cute.ceil_div(
                varlen_manager.len_k(batch_idx), self.cta_tile_shape_mnk[2]
            )
            for _k_tile in cutlass.range(k_tile_cnt, unroll=1):
                ab_pipeline.consumer_wait(landed_state)
                transpose(landed_state.index)
                bt_pipeline.producer_commit(transposed_state)
                landed_state.advance()
                transposed_state.advance()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _consume(
        self,
        ab_pipeline,
        bt_pipeline,
        epi_pipeline,
        tiled_mma: cute.TiledMma,
        mSFB: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        tma_atom_c: cute.CopyAtom | None,
        mC_mnl: cute.Tensor | None,
        epilogue_params,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sD: cute.Tensor,
        sC: cute.Tensor | None,
        epi_smem_tensors,
        varlen_manager: VarlenManager,
        TileSchedulerCls: Callable,
        warp_idx: Int32,
    ):
        """The MMA warpgroups: promoted mainloop, column order, epilogue."""
        has_C = const_expr(mC_mnl is not None)
        has_epi_load = const_expr(self.epi_c_stage > 0)
        is_tma_warp = Boolean(warp_idx == 0)
        tidx, _, _ = cute.arch.thread_idx()
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
        acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(
            thr_mma, self.cta_tile_shape_mnk, sA, sB
        )
        acc_sum = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
        mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)

        read_state = make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        epi_store_pipeline = self.make_epi_store_pipeline()
        epi_read_state = make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.epi_c_stage
        )
        epi_producer_state = make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.epi_c_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_coord_mnkl = work_tile.tile_idx
            batch_idx = tile_coord_mnkl[3]
            n_tile = tile_coord_mnkl[1]
            k_tile_cnt = cute.ceil_div(
                varlen_manager.len_k(batch_idx), self.cta_tile_shape_mnk[2]
            )
            # This tile's block scales per K tile (TILE_N == TILE_K == BLOCK).
            if const_expr(self.transpose_b):
                tile_scales = (mSFB[batch_idx, None, n_tile],)
            elif const_expr(self.concat_layout):
                # N tile t holds gate rows [64t, 64t + 64) and up rows
                # I + [64t, 64t + 64): with I a multiple of 64 (2I / 128 scale
                # rows), one scale block each.
                tile_scales = (
                    mSFB[batch_idx, n_tile // 2, None],
                    mSFB[batch_idx, (cute.size(mSFB, mode=[1]) + n_tile) // 2, None],
                )
            else:
                tile_scales = (mSFB[batch_idx, n_tile, None],)
            read_state = self.mma_promoted(
                ab_pipeline,
                bt_pipeline,
                read_state,
                mma_fn,
                acc,
                acc_sum,
                k_tile_cnt,
                tile_scales,
            )
            # The epilogue reads acc: the sum, its columns in the standard order.
            if const_expr(self.transpose_b):
                restore_column_order(acc_sum, acc)
            else:
                acc.store(acc_sum.load())

            copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                tma_atom_d,
                varlen_manager.offset_batch_epi(mD_mnl, batch_idx),
                self.cta_tile_shape_mnk[:2],
                self.epi_tile,
                sD,
                tile_coord_mnkl,
            )
            copy_C = None
            if const_expr(has_C):
                copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                    tma_atom_c,
                    varlen_manager.offset_batch_epi(mC_mnl, batch_idx),
                    self.cta_tile_shape_mnk[:2],
                    self.epi_tile,
                    sC,
                    tile_coord_mnkl,
                )
                copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)
            if const_expr(has_epi_load):
                tile_load_copy_fns = self.epi_tile_load_g2s_copy_fns(
                    epilogue_params,
                    epi_smem_tensors,
                    tile_coord_mnkl,
                    varlen_manager,
                    epi_pipeline,
                )
                copy_C = copy_utils.chain_tma_producer_copy_fns(
                    (copy_C, *tile_load_copy_fns)
                )
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                tiled_mma, self.d_layout, self.d_dtype, sD, tidx
            )
            tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
            load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
            tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = None, None, None, None
            if const_expr(has_C):
                tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = (
                    self.epilog_smem_load_and_partition(
                        tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                    )
                )
            self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)
            epi_read_state, epi_producer_state = self.epilogue(
                epilogue_params,
                epi_smem_tensors,
                epi_pipeline,
                epi_store_pipeline,
                epi_read_state,
                epi_producer_state,
                self.epi_tile,
                load_acc_subtile,
                tRS_rD,
                tRS_rC,
                None,
                tiled_copy_r2s,
                tRS_sD,
                tiled_copy_s2r,
                tSR_rC,
                tSR_sC,
                copy_D,
                copy_C,
                tile_coord_mnkl,
                varlen_manager,
                self.epilogue_barrier,
                tile_scheduler,
                tidx,
                is_tma_warp,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(self.use_pdl):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if is_tma_warp:
            epi_store_pipeline.producer_tail()

    @cute.jit
    def mma_promoted(
        self,
        ab_pipeline,
        bt_pipeline,
        read_state,
        mma_fn: Callable,
        acc: cute.Tensor,
        acc_sum: cute.Tensor,
        k_tile_cnt: Int32,
        tile_scales: tuple,
    ):
        """``acc_sum = sum_k scale[k] * (A_k @ B_k)`` over the tile's K tiles:
        each K tile's wgmmas start from zero in ``acc`` (FP8 accumulation
        never runs deeper than one 128-wide block) and are drained before the
        FP32 fold. With two scale vectors (gate/up rows interleaved), even
        columns take the first and odd columns the second: the accumulator
        fragment's fastest mode is the column pair, so slot ``i`` holds an
        even column exactly when ``i`` is even. Releases each stage (A and B)
        once its wgmmas retired."""
        paired = const_expr(len(tile_scales) == 2)
        even_scales = tile_scales[0]
        odd_scales = tile_scales[-1]
        acc_sum.fill(0.0)
        even_next = Float32(0.0)
        odd_next = Float32(0.0)
        if 0 < k_tile_cnt:
            even_next = even_scales[0]
            if const_expr(paired):
                odd_next = odd_scales[0]
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(read_state)
            if const_expr(self.transpose_b):
                bt_pipeline.consumer_wait(read_state)
            even = even_next
            odd = odd_next
            if k_tile + 1 < k_tile_cnt:
                even_next = even_scales[k_tile + 1]
                if const_expr(paired):
                    odd_next = odd_scales[k_tile + 1]
            mma_fn(A_idx=read_state.index, B_idx=read_state.index, zero_init=True)
            warpgroup.wait_group(0)
            if const_expr(paired):
                for i in cutlass.range_constexpr(cute.size(acc) // 2):
                    acc_sum[2 * i] = acc_sum[2 * i] + acc[2 * i] * even
                    acc_sum[2 * i + 1] = acc_sum[2 * i + 1] + acc[2 * i + 1] * odd
            else:
                acc_sum.store(acc_sum.load() + acc.load() * even)
            ab_pipeline.consumer_release(read_state)
            read_state.advance()
        return read_state


def column_source(c, j, g):
    """Where the accumulator value at fragment slot ``(c, j, g)`` (column pair
    element, row half, 8-column group) comes from before `restore_column_order`:
    the transposed tile's N order swaps slot ``(1, j, 2i)`` with
    ``(0, j, 2i + 1)`` (FlashAttention-3's ``permute_output_fp8``)."""
    return g & 1, j, (g & ~1) | c


@cute.jit
def restore_column_order(acc_sum: cute.Tensor, acc: cute.Tensor):
    """``acc`` = ``acc_sum`` with its columns in the standard order (static
    register moves within each thread)."""
    groups = acc_sum.shape[0][2]
    for m in cutlass.range_constexpr(cute.size(acc_sum, mode=[1])):
        for n in cutlass.range_constexpr(cute.size(acc_sum, mode=[2])):
            for g in cutlass.range_constexpr(groups):
                for j in cutlass.range_constexpr(2):
                    for c in cutlass.range_constexpr(2):
                        acc[(c, j, g), m, n] = acc_sum[column_source(c, j, g), m, n]


class RowScaleEpiMixin(ComposableEpiMixin):
    """Epilogue ``D = acc * xs (+ C)``: one FP32 scale per output row
    (``[L, M]`` dense, ``[total M]`` varlen), plus the C operand when given."""

    _epi_ops = (ColVecLoad("xs"),)

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        xs: cute.Tensor | None = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = RoundingMode.RN
        return self.EpilogueParams(**self._epi_ops_to_params_dict(args))

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        row_scale = epi_loop_tensors.get("xs")
        if const_expr(row_scale is not None):
            for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
                tRS_rD[i] = tRS_rD[i] * row_scale[i]
        if const_expr(tRS_rC is not None):
            tRS_rD.store(tRS_rD.load() + tRS_rC.load().to(tRS_rD.element_type))
        return ()


class GemmSm90BlockFP8TRowScale(RowScaleEpiMixin, GemmSm90BlockFP8T):
    """`GemmSm90BlockFP8T` storing ``acc * xs (+ C)``."""


# ---- fused epilogues -------------------------------------------------------------

_ROW_SCALE = {"xs": ColVecLoad("xs")}
_LIMIT = {"limit": Scalar("limit")}


@gemm_epilogue(outputs=("postact",), mode="acc_pair", ops={**_ROW_SCALE, **_LIMIT})
def row_scaled_swiglu(acc, xs, limit):
    value = acc * xs
    gate, up = unpack(value)
    return {"D": value, "postact": swiglu_clamped(gate, up, limit)}


@gemm_epilogue(outputs=("postact",), mode="acc_pair", ops={**_ROW_SCALE, **_LIMIT})
def row_scaled_swiglu_add(acc, c, xs, limit):
    value = acc * xs + c
    gate, up = unpack(value)
    return {"D": value, "postact": swiglu_clamped(gate, up, limit)}


@gemm_epilogue(
    outputs=("postact",),
    mode="packed_cd_b16x2",
    ops={**_ROW_SCALE, "score": ColVecLoad("score"), **_LIMIT},
    reduces={"dscore": ColVecReduce("dscore", scaled=True)},
)
def row_scaled_dswiglu_scored(acc, c, xs, score, limit):
    gate, up = unpack(c)
    r = acc * xs
    dg, du, activation = dswiglu_clamped(gate, up, r * score, limit)
    return {"D": pack(dg, du), "postact": activation * score, "dscore": (activation, r)}


@functools.cache
def epilogue_gemm_class(epilogue):
    """`GemmSm90BlockFP8T` with a `gemm_epilogue` function's epilogue: the
    class Quack's frontend mints for SM90, on this kernel."""
    visit = tuple(
        ("c", "c") if name == "c" else (name, _pinned_visit_kind(epilogue.ops[name]))
        for name in epilogue.operand_names
    )
    minted = epilogue._mint(
        visit,
        9,
        epilogue.mode == "acc_pair",
        epilogue.mode == "packed_cd_b16x2",
        (),
        RoundingMode.RN,
        (),
        False,
    )
    body = {
        key: value
        for key, value in vars(minted).items()
        if key not in ("__dict__", "__weakref__")
    }
    name = f"{minted.__name__}_BlockFP8T"
    body["__qualname__"] = name
    return type(name, (_EpiModMixinBase, GemmSm90BlockFP8T), body)


# ---- host side -----------------------------------------------------------------


def _epilogue_keys(gemm_cls, values):
    """Each epilogue op's compile-time key for its value (None: absent)."""
    return tuple(
        sorted(
            (op.name, op.host_arg_key(values.get(op.name))) for op in gemm_cls._epi_ops
        )
    )


@functools.cache
def compile_gemm(
    gemm_cls,
    epilogue_keys,
    *,
    transpose_b,
    varlen,
    concat=False,
    c_dtype=None,
    packed_cd=False,
    cluster_m=1,
    arch=None,
    options=None,
):
    """``gemm_cls`` (`GemmSm90BlockFP8TRowScale` or an `epilogue_gemm_class`)
    compiled for one mode: shapes are symbolic. ``epilogue_keys`` from
    `_epilogue_keys`; ``arch`` (e.g. ``"sm_90a"``) pins the target, else the
    device's is used; ``options`` as for `compile_tile_transpose`."""
    gemm = gemm_cls(
        Float32,
        cutlass.Float8E4M3FN,
        (TILE_M, TILE_N, TILE_K),
        (cluster_m, 1, 1),
        concat_layout=("B",) if concat else None,
        transpose_b=transpose_b,
    )
    # The transposed GEMM's B crosses as the stored weight (l, k, n); the
    # trace reads it as (n, k, l).
    gemm.b_transposed = transpose_b
    if packed_cd:
        # C and D cross as their 16-bit pairs; the trace reads them as FP32.
        gemm.cd_packed = "n"
        gemm.implicit_dtype = c_dtype
    fp8 = cutlass.Float8E4M3FN
    mA, mB, mD, mC, m, n, k, l = make_fake_gemm_tensors(
        fp8,
        fp8,
        c_dtype if packed_cd else cutlass.BFloat16,
        c_dtype,
        "k",
        "n" if transpose_b else "k",
        "n",
        "n",
        varlen_m=varlen,
        batched=True,
        b_kn=transpose_b,
        packed_cd="n" if packed_cd else None,
    )
    context = FakeArgCtx(m, n, k, l, not varlen, varlen)
    ops = {op.name: op for op in gemm_cls._epi_ops}
    fields = {}
    for name, key in epilogue_keys:
        fake = ops[name].host_fake_arg(key, context)
        if fake is not None:
            fields[name] = fake
    scale = make_fake_tensor(
        Float32, (l, cute.sym_int(), cute.sym_int()), leading_dim=2
    )
    return cute.compile(
        gemm,
        mA,
        mB,
        mD,
        mC,
        gemm_cls.EpilogueArguments(**fields),
        make_fake_scheduler_args(False, False, l),
        make_fake_varlen_args(True, False, False, None) if varlen else None,
        make_fake_stream(),
        None,
        scale,
        options=_options(arch, options),
    )


def _check(name, tensor, dtype, ndim):
    if tensor.dtype != dtype or tensor.ndim != ndim:
        raise ValueError(f"{name}: expected a {ndim}-D {dtype} tensor")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous along its last dimension")


def _launch(
    gemm_cls,
    a,
    a_scale,
    weight,
    weight_scale,
    out,
    *,
    offsets,
    c=None,
    values=None,
    transpose_b,
    concat=False,
    packed_cd=False,
    cluster_m=1,
):
    """Validate the common operands and run ``gemm_cls`` (the epilogue's
    other values in ``values``)."""
    varlen = offsets is not None
    _check("weight", weight, torch.float8_e4m3fn, 3)
    _check("a", a, torch.float8_e4m3fn, 2 if varlen else 3)
    _check("a_scale", a_scale, torch.float32, 1 if varlen else 2)
    experts, rows_w, cols_w = weight.shape
    k = rows_w if transpose_b else cols_w
    if k % 16 or rows_w % 16 or cols_w % 16:
        raise ValueError("the weight's dimensions must be multiples of 16")
    blocks = (experts, -(-rows_w // BLOCK), -(-cols_w // BLOCK))
    if weight_scale.dtype != torch.float32 or tuple(weight_scale.shape) != blocks:
        raise ValueError(f"weight_scale: expected FP32 {blocks}")
    if not weight_scale.is_contiguous():
        raise ValueError("weight_scale must be contiguous")
    if a.shape[-1] != k:
        raise ValueError(f"a: expected {k} columns")
    if tuple(a_scale.shape) != tuple(a.shape[:-1]):
        raise ValueError("a_scale: expected one scale per row of a")
    if varlen:
        if offsets.dtype != torch.int32 or offsets.shape != (experts + 1,):
            raise ValueError(f"offsets: expected int32 [{experts + 1}]")
        if offsets.data_ptr() % 16:
            raise ValueError("offsets must be 16-byte aligned")
    elif a.shape[0] != experts:
        raise ValueError("a: expected one batch per expert")
    if c is not None and (c.shape != out.shape or c.stride(-1) != 1):
        raise ValueError(f"c: expected {tuple(out.shape)}, contiguous along columns")
    values = {"xs": a_scale, **(values or {})}
    keys = _epilogue_keys(gemm_cls, values)
    compiled = compile_gemm(
        gemm_cls,
        keys,
        transpose_b=transpose_b,
        varlen=varlen,
        concat=concat,
        c_dtype=torch2cute_dtype_map[c.dtype] if c is not None else None,
        packed_cd=packed_cd,
        cluster_m=cluster_m,
    )
    ops = {op.name: op for op in gemm_cls._epi_ops}
    fields = {
        name: ops[name].host_call_arg(values[name], key)
        for name, key in keys
        if key is not None and values.get(name) is not None
    }
    # The launch runs on the current stream (TVM-FFI's environment stream), so
    # the call passes every compiled argument but the stream.
    compiled(
        a,
        weight,
        out,
        c,
        gemm_cls.EpilogueArguments(**fields),
        make_scheduler_args(get_max_active_clusters(cluster_m), 8, None),
        make_varlen_args(offsets, None, None) if varlen else None,
        None,  # mSFA
        weight_scale,  # mSFB
    )


def _rows_out(a, n, dtype=torch.bfloat16):
    return torch.empty(*a.shape[:-1], n, dtype=dtype, device=a.device)


def block_fp8_transposed_gemm(
    a, a_scale, weight, weight_scale, *, offsets=None, out=None, cluster_m=1
):
    """``out = (a * a_scale) @ (weight * weight_scale)`` per expert, in bf16.

    ``weight [E, K, N]`` E4M3 as stored (N contiguous), ``weight_scale [E,
    ceil(K/128), ceil(N/128)]`` FP32. ``a`` E4M3 with its columns in the
    transposed GEMM's K order (``quantize_rows_fp8(..., transposed_gemm=True)``)
    and ``a_scale`` FP32 per row. Dense: ``a [E, M, K]``, ``a_scale [E, M]``,
    ``out [E, M, N]``. Varlen: ``a [R, K]``, ``a_scale [R]``, ``offsets
    [E + 1]`` int32 bounding each expert's rows, ``out [R, N]``. K and N are
    multiples of 16."""
    if out is None:
        out = _rows_out(a, weight.shape[2])
    elif out.dtype != torch.bfloat16 or tuple(out.shape) != (
        *a.shape[:-1],
        weight.shape[2],
    ):
        raise ValueError("out: expected bf16 rows of a by N")
    if out.numel():
        _launch(
            GemmSm90BlockFP8TRowScale,
            a,
            a_scale,
            weight,
            weight_scale,
            out,
            offsets=offsets,
            transpose_b=True,
            cluster_m=cluster_m,
        )
    return out


def block_fp8_gemm(
    a, a_scale, weight, weight_scale, *, offsets=None, c=None, out=None, cluster_m=1
):
    """``out = (a * a_scale) @ (weight * weight_scale)ᵀ (+ c)`` per expert, in
    bf16: a projection through a weight stored ``[E, N, K]`` (K contiguous),
    ``weight_scale [E, ceil(N/128), ceil(K/128)]`` FP32; ``a`` E4M3 rows in
    natural order with FP32 ``a_scale`` per row, shaped as for
    `block_fp8_transposed_gemm`; ``c`` (bf16, shaped as ``out``) added."""
    if out is None:
        out = _rows_out(a, weight.shape[1])
    if out.numel():
        _launch(
            GemmSm90BlockFP8TRowScale,
            a,
            a_scale,
            weight,
            weight_scale,
            out,
            offsets=offsets,
            c=c,
            transpose_b=False,
            cluster_m=cluster_m,
        )
    return out


def block_fp8_gemm_swiglu(
    a, a_scale, weight, weight_scale, *, offsets=None, c=None, limit=None, cluster_m=1
):
    """``(h, act)`` of the up projection through ``weight [E, 2I, K]`` (gate
    rows then up rows, K contiguous; block scales ``[E, 2I/128, K/128]``):
    ``h = (a * a_scale) @ weightᵀ (+ c)`` with gate and up interleaved
    (``h[:, 2j]`` gate ``j``, ``h[:, 2j + 1]`` up ``j``; ``c`` in the same
    order), and ``act = silu(min(gate, limit)) * clamp(up, -limit, limit)``
    (no clamp when ``limit`` is None). ``I`` is a multiple of 64."""
    two_i = weight.shape[1]
    if two_i % BLOCK:
        raise ValueError("the gate and up halves must be multiples of 64 rows")
    h = _rows_out(a, two_i)
    act = _rows_out(a, two_i // 2)
    if h.numel():
        epilogue = row_scaled_swiglu_add if c is not None else row_scaled_swiglu
        _launch(
            epilogue_gemm_class(epilogue),
            a,
            a_scale,
            weight,
            weight_scale,
            h,
            offsets=offsets,
            c=c,
            values={
                "limit": math.inf if limit is None else float(limit),
                "postact": act,
            },
            transpose_b=False,
            concat=True,
            cluster_m=cluster_m,
        )
    return h, act


def block_fp8_transposed_gemm_dswiglu(
    a, a_scale, weight, weight_scale, h, score, *, offsets=None, limit=None, cluster_m=1
):
    """The down projection's input gradient with the SwiGLU backward fused:
    ``r = (a * a_scale) @ (weight * weight_scale)`` as in
    `block_fp8_transposed_gemm` (``weight [E, H, I]`` as stored), then from the
    interleaved pre-activation ``h [R, 2I]`` (bf16) and the per-row ``score``:
    ``(dh, act * score, dscore)``, ``dh`` the interleaved gradient of
    ``score * r`` through the SwiGLU (clamped at ``limit`` when given), and
    ``dscore = <act, r>`` per row (FP32)."""
    i = weight.shape[2]
    if h.dtype != torch.bfloat16 or tuple(h.shape) != (*a.shape[:-1], 2 * i):
        raise ValueError(f"h: expected bf16 rows of a by {2 * i}")
    if score.dtype != torch.float32 or tuple(score.shape) != tuple(a.shape[:-1]):
        raise ValueError("score: expected one FP32 score per row of a")
    gemm_cls = epilogue_gemm_class(row_scaled_dswiglu_scored)
    dh = torch.empty_like(h)
    act = _rows_out(a, i)
    sink = row_scaled_dswiglu_scored.sinks["dscore"]
    partials = torch.empty(
        sink.sink_alloc_shape(tuple(a.shape[:-1]), i, TILE_M, TILE_N),
        dtype=torch.float32,
        device=a.device,
    )
    if dh.numel():
        _launch(
            gemm_cls,
            a,
            a_scale,
            weight,
            weight_scale,
            dh,
            offsets=offsets,
            c=h,
            values={
                "score": score,
                "limit": math.inf if limit is None else float(limit),
                "postact": act,
                "dscore": partials,
            },
            transpose_b=True,
            packed_cd=True,
            cluster_m=cluster_m,
        )
    return dh, act, sink.host_finalize(partials)


__all__ = [
    "BLOCK",
    "K_ORDER",
    "N_ORDER",
    "GemmSm90BlockFP8T",
    "GemmSm90BlockFP8TRowScale",
    "RowScaleEpiMixin",
    "TileTransposeKernel",
    "block_fp8_gemm",
    "block_fp8_gemm_swiglu",
    "block_fp8_transposed_gemm",
    "block_fp8_transposed_gemm_dswiglu",
    "column_source",
    "compile_gemm",
    "compile_tile_transpose",
    "epilogue_gemm_class",
    "k_major_view",
    "make_tile_transposer",
    "mn_major_view",
    "restore_column_order",
    "transpose_copies",
    "transposed_tiles",
]
