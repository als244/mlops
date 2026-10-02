"""FP8 activation payloads over stock MoonEP, with explicit gradient inputs.

The caller owns a full-width BF16 buffer used by combine. Its borrowed dispatch
view copies FP8 bytes unchanged. Weight gradients use the values represented by
the received FP8 payload and descales. No original BF16 rows are dispatched.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from moonep import Buffer
from moonep.inter_rank_sync import launch_inter_rank_sync

from .kernels.quantize_rows import quantize_rows_fp8


@dataclass
class FP8Rows:
    data: torch.Tensor
    scales: torch.Tensor


def row_data(value):
    return value.data if isinstance(value, FP8Rows) else value


def quantized_rows(value):
    if isinstance(value, FP8Rows):
        return value.data, value.scales
    return quantize_rows_fp8(value)


def gradient_rows(value):
    if isinstance(value, FP8Rows):
        return value.data, value.scales
    return value, None


def mask_rows_tail(pointwise, value, end):
    pointwise.mask_tail(row_data(value), end)
    if isinstance(value, FP8Rows):
        pointwise.mask_tail(value.scales, end)


def save_rows(value):
    """Return only ordinary tensors for the custom operator's saved-state ABI."""
    data, scales = gradient_rows(value)
    return data, [] if scales is None else [scales]


def restore_rows(data, extra):
    return FP8Rows(data, extra[0]) if extra else data


def empty_bf16_rows(value):
    data = row_data(value)
    return torch.empty(data.shape, device=data.device, dtype=torch.bfloat16)


@triton.jit
def _scatter_scale(
    SCALE,
    DST,
    META,
    N: tl.constexpr,
    K: tl.constexpr,
    NVS: tl.constexpr,
    STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    destination = tl.load(DST + index, index < N, 0)
    raw = tl.where(destination >= 0, destination, -destination - 1).to(tl.int64)
    value = tl.load(SCALE + index // K, index < N, 0)
    address = (raw // NVS) * STRIDE + OFFSET + raw % NVS
    tl.store(META + address, value.to(tl.int32, bitcast=True), index < N)


class FP8DispatchView(Buffer):
    """Borrow the owner's allocations while interpreting hidden rows at half width."""

    def __init__(self, owner):
        if not isinstance(owner, Buffer) or isinstance(owner, FP8DispatchView):
            raise TypeError(
                "FP8 dispatch requires a compatible caller-owned MoonEP Buffer"
            )
        original = owner._require_ctx()
        required = {
            "H",
            "NvS_padded",
            "NvS",
            "R",
            "rank",
            "hidden_buf",
            "meta_buf",
            "meta_chunk_padded",
            "WEIGHTS_OFF",
            "S",
            "K",
            "N",
        }
        if not required <= original.keys():
            raise ValueError(
                "FP8 dispatch is incompatible with this MoonEP buffer descriptor"
            )
        hidden = original["hidden_buf"]
        metadata = original["meta_buf"]
        if (
            original["H"] % 16
            or hidden.dtype != torch.bfloat16
            or not hidden.is_cuda
            or not hidden.is_contiguous()
            or hidden.numel() != original["R"] * original["NvS_padded"] * original["H"]
        ):
            raise ValueError(
                "FP8 dispatch requires contiguous BF16 MoonEP storage with "
                "a 16-byte-aligned feature width and the expected rank pitch"
            )
        if (
            metadata.dtype != torch.int32
            or not metadata.is_contiguous()
            or metadata.device != hidden.device
        ):
            raise ValueError(
                "FP8 dispatch requires contiguous INT32 MoonEP metadata on the buffer device"
            )
        self.owner = owner
        self._ctx = dict(original)
        ctx = self._ctx
        ctx["H"] //= 2
        ctx["NvS_padded"] *= 2
        ctx["hidden_buf"] = original["hidden_buf"].view(
            ctx["R"] * ctx["NvS_padded"], ctx["H"]
        )
        begin = ctx["rank"] * ctx["NvS_padded"]
        ctx["hidden_buf_local"] = ctx["hidden_buf"][begin : begin + ctx["NvS"]]
        self._comm_stream = owner._comm_stream
        self.enable_pdl = owner.enable_pdl
        self._destroyed = False

    def _require_ctx(self):
        if self.owner.destroyed:
            raise RuntimeError("FP8 dispatch owner has been destroyed")
        return self._ctx

    def __del__(self):
        # Never destroy borrowed VMM allocations or synchronize during collection.
        pass

    def destroy(self):
        raise RuntimeError(
            "Destroy the caller-owned MoonEP buffer, not its dispatch view"
        )

    def dispatch_activation(
        self,
        value,
        probabilities=None,
        ids=None,
        histogram=None,
        *,
        plan=None,
        zero_copy=False,
    ):
        payload, scales = quantize_rows_fp8(value)
        received, descales, pp, ends, used_plan = self.dispatch_rows(
            payload,
            scales,
            probabilities,
            ids,
            histogram,
            plan=plan,
            zero_copy=zero_copy,
        )
        return FP8Rows(received, descales), pp, ends, used_plan

    def dispatch_rows(
        self,
        data,
        scales,
        probabilities=None,
        ids=None,
        histogram=None,
        *,
        plan=None,
        zero_copy=False,
    ):
        """Use the current stream, returning (data, scales, probabilities, ends, plan).

        Forward preserves routing probabilities separately before staging scales.
        Backward plan reuse sends scales through the ordinary route-weight channel.
        All ranks must execute the same calls on the communication stream.
        """
        ctx = self._require_ctx()
        if data.dtype != torch.float8_e4m3fn or not data.is_contiguous():
            raise ValueError("Expected a contiguous token-major E4M3 payload")
        if data.shape != (ctx["S"], 2 * ctx["H"]):
            raise ValueError("FP8 payload shape disagrees with the owner geometry")
        if (
            scales.shape != (ctx["S"],)
            or scales.dtype != torch.float32
            or not scales.is_contiguous()
        ):
            raise ValueError("Expected the original contiguous FP32 token descales")
        if plan is not None and probabilities is not None:
            raise ValueError(
                "Plan reuse expects saved routing probabilities, not a new probability input"
            )
        wire = data.view(torch.bfloat16)
        if plan is not None:
            rows = scales[:, None].expand(-1, ctx["K"]).contiguous()
            received, received_scales, ends, plan = super().dispatch(
                wire,
                rows,
                plan=plan,
                async_finish=False,
                inter_rank_sync=True,
                zero_copy=zero_copy,
                router_weights_zero_copy=False,
            )
            return received.view(data.dtype), received_scales, None, ends, plan
        if probabilities is None:
            raise ValueError("Fresh planning needs routing probabilities")
        received, received_probabilities, ends, plan = super().dispatch(
            wire,
            probabilities,
            ids,
            histogram,
            async_finish=False,
            inter_rank_sync=True,
            zero_copy=zero_copy,
            router_weights_zero_copy=False,
        )
        # Remote copies of routing probabilities must finish before this rank
        # overwrites their communication slots with descales.
        launch_inter_rank_sync(ctx)
        _scatter_scale[(triton.cdiv(ctx["N"], 256),)](
            scales,
            plan.dst,
            ctx["meta_buf"],
            ctx["N"],
            ctx["K"],
            ctx["NvS"],
            ctx["meta_chunk_padded"],
            ctx["WEIGHTS_OFF"],
            256,
        )
        launch_inter_rank_sync(ctx)
        received_scales = self.router_weight_buffer_view.clone()
        return (
            received.view(data.dtype),
            received_scales,
            received_probabilities,
            ends,
            plan,
        )
