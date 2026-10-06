"""Batch expert-column quantization without changing per-expert scale domains.

Three launches cover every group: partial maxima, scales, transpose/cast.
Each feature scale is reduced once, and inactive token tiles exit immediately.
Group lengths must have a 16-byte FP8 pitch, with 128-row MoonEP padding by
default. The caller retains exact group lengths for its per-expert GEMMs.
"""

import itertools

import torch
import triton
import triton.language as tl


@triton.jit
def _group_amax(
    X,
    CU,
    P,
    ROW_S,
    F: tl.constexpr,
    XS: tl.constexpr,
    BT: tl.constexpr,
    BF: tl.constexpr,
    FROM_FP8: tl.constexpr,
):
    group = tl.program_id(2)
    begin, end = tl.load(CU + group).to(tl.int64), tl.load(CU + group + 1).to(tl.int64)
    if begin + tl.program_id(0) * BT < end:
        token = tl.program_id(0) * BT + tl.arange(0, BT)
        feature = tl.program_id(1) * BF + tl.arange(0, BF)
        value = tl.load(
            X + (begin + token[:, None]) * XS + feature[None, :],
            (token[:, None] < end - begin) & (feature[None, :] < F),
            0.0,
        ).to(tl.float32)
        if FROM_FP8:
            descale = tl.load(ROW_S + begin + token, token < end - begin, 0)
            value = (value * descale[:, None]).to(tl.bfloat16).to(tl.float32)
        maximum = tl.max(tl.abs(value), axis=0)
        partial_base = begin // BT + group
        tl.store(
            P + (partial_base + tl.program_id(0)) * F + feature, maximum, feature < F
        )


@triton.jit
def _group_scales(
    CU, P, S, F: tl.constexpr, BT: tl.constexpr, BP: tl.constexpr, BF: tl.constexpr
):
    group = tl.program_id(1)
    begin, end = tl.load(CU + group), tl.load(CU + group + 1)
    if begin < end:
        feature = tl.program_id(0) * BF + tl.arange(0, BF)
        part = tl.arange(0, BP)
        partial_base = begin // BT + group
        parts = tl.cdiv(end - begin, BT)
        partial = tl.load(
            P + (partial_base + part[:, None]) * F + feature[None, :],
            (part[:, None] < parts) & (feature[None, :] < F),
            0,
        )
        scale = tl.maximum(tl.max(partial, axis=0) * (1.0 / 448.0), 1e-12)
        tl.store(S + group * F + feature, scale, feature < F)


@triton.jit
def _group_cast(
    X,
    CU,
    S,
    Q,
    ROW_S,
    F: tl.constexpr,
    XS: tl.constexpr,
    BT: tl.constexpr,
    BF: tl.constexpr,
    FROM_FP8: tl.constexpr,
):
    group = tl.program_id(2)
    begin, end = tl.load(CU + group).to(tl.int64), tl.load(CU + group + 1).to(tl.int64)
    if begin + tl.program_id(0) * BT < end:
        token = tl.program_id(0) * BT + tl.arange(0, BT)
        feature = tl.program_id(1) * BF + tl.arange(0, BF)
        scale = tl.load(S + group * F + feature, feature < F, 1)
        value = tl.load(
            X + (begin + token[:, None]) * XS + feature[None, :],
            (token[:, None] < end - begin) & (feature[None, :] < F),
            0.0,
        ).to(tl.float32)
        if FROM_FP8:
            descale = tl.load(ROW_S + begin + token, token < end - begin, 0)
            value = (value * descale[:, None]).to(tl.bfloat16).to(tl.float32)
        quantized = tl.minimum(
            tl.maximum(tl.div_rn(value, scale[None, :]), -448.0), 448.0
        )
        tl.store(
            Q + begin * F + feature[:, None] * (end - begin) + token[None, :],
            tl.trans(quantized.to(Q.dtype.element_ty)),
            (feature[:, None] < F) & (token[None, :] < end - begin),
        )


@triton.jit
def _group_slots(IDS, OUT, N: tl.constexpr, Q: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    expert = tl.load(IDS + i, i < N, -1)
    slot = tl.where(expert >= 0, expert + (expert // Q) * Q, -1)
    tl.store(OUT + i, slot, i < N)


def group_slots(ids, local_experts):
    out = torch.empty_like(ids, dtype=torch.int32)
    _group_slots[(1,)](
        ids,
        out,
        ids.numel(),
        local_experts,
        triton.next_power_of_2(ids.numel()),
        num_warps=4,
    )
    return out


def quantize_groups(value, cu, offsets, *, row_scales=None):
    """Return packed feature-major FP8 groups and [group, feature] FP32 scales."""
    from_fp8 = value.dtype == torch.float8_e4m3fn
    if (
        value.ndim != 2
        or value.stride(1) != 1
        or value.dtype not in (torch.bfloat16, torch.float32, torch.float8_e4m3fn)
    ):
        raise ValueError(
            "Grouped quantization requires a BF16/FP32/FP8 matrix contiguous along features"
        )
    if from_fp8:
        if (
            row_scales is None
            or row_scales.shape != value.shape[:1]
            or row_scales.dtype != torch.float32
            or row_scales.device != value.device
            or not row_scales.is_contiguous()
        ):
            raise ValueError(
                "FP8 input requires contiguous FP32 row descales on the same device"
            )
    elif row_scales is not None:
        raise ValueError("Row descales are only valid for FP8 input")
    scale_input = row_scales if from_fp8 else value
    lengths = [b - a for a, b in itertools.pairwise(offsets)]
    if not lengths or min(lengths) < 0 or any(n % 16 for n in lengths):
        raise ValueError(
            "Grouped FP8 quantization requires nonnegative 16-aligned group lengths"
        )
    if offsets[0] != 0 or offsets[-1] > value.shape[0] or cu.numel() != len(offsets):
        raise ValueError("Group offsets do not describe the input matrix")
    features = value.shape[1]
    data = torch.empty(
        value.shape[0] * features, device=value.device, dtype=torch.float8_e4m3fn
    )
    scales = torch.empty(
        (len(lengths), features), device=value.device, dtype=torch.float32
    )
    # Each group's tile range begins at floor(begin / 256) + group. One
    # additional slot per group covers its partial tile without overlap.
    # Capacity depends only on input geometry, never the routing distribution.
    partial = torch.empty(
        (triton.cdiv(value.shape[0], 256) + len(lengths), features),
        device=value.device,
        dtype=torch.float32,
    )
    maximum = max(lengths)
    if maximum:
        parts = triton.cdiv(maximum, 256)
        _group_amax[(parts, triton.cdiv(features, 64), len(lengths))](
            value,
            cu,
            partial,
            scale_input,
            features,
            value.stride(0),
            256,
            64,
            from_fp8,
            num_warps=4,
        )
        _group_scales[(triton.cdiv(features, 128), len(lengths))](
            cu,
            partial,
            scales,
            features,
            256,
            triton.next_power_of_2(parts),
            128,
            num_warps=4,
        )
        _group_cast[
            (triton.cdiv(maximum, 64), triton.cdiv(features, 64), len(lengths))
        ](
            value,
            cu,
            scales,
            data,
            scale_input,
            features,
            value.stride(0),
            64,
            64,
            from_fp8,
            num_warps=4,
        )
    return data, scales
