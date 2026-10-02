"""Per-feature E4M3 quantization directly from token-major BF16/FP32 storage.

This preserves QuACK's per-row quantization of ``value.T`` exactly, but never
materializes that full-precision transpose. Partial maxima are small FP32
scratch; the final tiled pass converts and transposes into the FP8 destination.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _load_values(
    X,
    ROW_S,
    token,
    feature,
    T: tl.constexpr,
    F: tl.constexpr,
    XS: tl.constexpr,
    FROM_FP8: tl.constexpr,
):
    value = tl.load(
        X + token[:, None] * XS + feature[None, :],
        (token[:, None] < T) & (feature[None, :] < F),
        0.0,
    ).to(tl.float32)
    if FROM_FP8:
        descale = tl.load(ROW_S + token, token < T, 0)
        # This recipe quantizes the BF16 values represented by dispatched FP8.
        # Round here in both passes, without allocating a BF16 intermediate.
        value = (value * descale[:, None]).to(tl.bfloat16).to(tl.float32)
    return value


@triton.jit
def _column_amax(
    X,
    P,
    ROW_S,
    T: tl.constexpr,
    F: tl.constexpr,
    XS: tl.constexpr,
    BT: tl.constexpr,
    BF: tl.constexpr,
    FROM_FP8: tl.constexpr,
):
    token = tl.program_id(0) * BT + tl.arange(0, BT)
    feature = tl.program_id(1) * BF + tl.arange(0, BF)
    value = _load_values(X, ROW_S, token, feature, T, F, XS, FROM_FP8)
    amax = tl.max(tl.abs(value), axis=0)
    tl.store(P + tl.program_id(0) * F + feature, amax, feature < F)


@triton.jit
def _column_scales(
    P, S, F: tl.constexpr, PARTS: tl.constexpr, BP: tl.constexpr, BF: tl.constexpr
):
    part = tl.arange(0, BP)
    feature = tl.program_id(0) * BF + tl.arange(0, BF)
    value = tl.load(
        P + part[:, None] * F + feature[None, :],
        (part[:, None] < PARTS) & (feature[None, :] < F),
        0,
    )
    scale = tl.maximum(tl.max(value, axis=0) * (1.0 / 448.0), 1e-12)
    tl.store(S + feature, scale, feature < F)


@triton.jit
def _cast_transpose(
    X,
    Q,
    S,
    ROW_S,
    T: tl.constexpr,
    F: tl.constexpr,
    XS: tl.constexpr,
    QS: tl.constexpr,
    BT: tl.constexpr,
    BF: tl.constexpr,
    FROM_FP8: tl.constexpr,
):
    token = tl.program_id(0) * BT + tl.arange(0, BT)
    feature = tl.program_id(1) * BF + tl.arange(0, BF)
    value = _load_values(X, ROW_S, token, feature, T, F, XS, FROM_FP8)
    scale = tl.load(S + feature, feature < F, 1)
    value = tl.minimum(tl.maximum(tl.div_rn(value, scale[None, :]), -448.0), 448.0)
    # Convert before exchanging layout: the transpose moves FP8 values.
    quantized = value.to(Q.dtype.element_ty)
    tl.store(
        Q + feature[:, None] * QS + token[None, :],
        tl.trans(quantized),
        (feature[:, None] < F) & (token[None, :] < T),
    )


@triton.jit
def _small_quantize_transpose(
    X,
    Q,
    S,
    ROW_S,
    T: tl.constexpr,
    F: tl.constexpr,
    XS: tl.constexpr,
    QS: tl.constexpr,
    BT: tl.constexpr,
    BF: tl.constexpr,
    FROM_FP8: tl.constexpr,
):
    token = tl.arange(0, BT)
    feature = tl.program_id(0) * BF + tl.arange(0, BF)
    value = _load_values(X, ROW_S, token, feature, T, F, XS, FROM_FP8)
    scale = tl.maximum(tl.max(tl.abs(value), axis=0) * (1.0 / 448.0), 1e-12)
    quantized = tl.minimum(tl.maximum(tl.div_rn(value, scale[None, :]), -448.0), 448.0)
    tl.store(
        Q + feature[:, None] * QS + token[None, :],
        tl.trans(quantized.to(Q.dtype.element_ty)),
        (feature[:, None] < F) & (token[None, :] < T),
    )
    tl.store(S + feature, scale, feature < F)


def quantize_transpose_fp8(value, *, poison_padding=False, row_scales=None):
    """Return FP8 [features, tokens] and FP32 [features] descales.

    A column's scale is ``max(max(abs(x[:, f])) / 448, 1e-12)``. The FP32
    reciprocal rounding, RN division, clipping, and E4M3 rounding match the
    existing row quantizer. Only the physical FP8 pitch is aligned to 16 bytes;
    the logical token count is unchanged and padded bytes are never operands.
    """
    if value.ndim != 2 or value.stride(1) != 1 or not value.is_cuda:
        raise ValueError("Expected a CUDA matrix contiguous along features")
    from_fp8 = row_scales is not None
    if from_fp8:
        if (
            value.dtype != torch.float8_e4m3fn
            or row_scales.dtype != torch.float32
            or row_scales.shape != (value.shape[0],)
            or not row_scales.is_contiguous()
            or row_scales.device != value.device
        ):
            raise ValueError("Expected E4M3 rows and contiguous FP32 token descales")
    elif value.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Expected BF16 or FP32 source values, or FP8 plus row_scales")
    scale_input = row_scales if from_fp8 else value
    tokens, features = value.shape
    if tokens == 0:
        raise ValueError("Cannot quantize an empty reduction dimension")
    pitch = triton.cdiv(tokens, 16) * 16
    data = torch.empty(
        (features, pitch), device=value.device, dtype=torch.float8_e4m3fn
    )
    scales = torch.empty(features, device=value.device, dtype=torch.float32)
    if poison_padding:
        data.view(torch.uint8).fill_(0x7F)
    if features:
        if tokens <= 256:
            _small_quantize_transpose[(triton.cdiv(features, 64),)](
                value,
                data,
                scales,
                scale_input,
                tokens,
                features,
                value.stride(0),
                pitch,
                triton.next_power_of_2(tokens),
                64,
                from_fp8,
                num_warps=4,
            )
        else:
            parts = triton.cdiv(tokens, 256)
            partial = torch.empty(
                (parts, features), device=value.device, dtype=torch.float32
            )
            _column_amax[(parts, triton.cdiv(features, 64))](
                value,
                partial,
                scale_input,
                tokens,
                features,
                value.stride(0),
                256,
                64,
                from_fp8,
                num_warps=4,
            )
            _column_scales[(triton.cdiv(features, 128),)](
                partial,
                scales,
                features,
                parts,
                triton.next_power_of_2(parts),
                128,
                num_warps=4,
            )
            _cast_transpose[(triton.cdiv(tokens, 64), triton.cdiv(features, 64))](
                value,
                data,
                scales,
                scale_input,
                tokens,
                features,
                value.stride(0),
                pitch,
                64,
                64,
                from_fp8,
                num_warps=4,
            )
    return data[:, :tokens], scales
