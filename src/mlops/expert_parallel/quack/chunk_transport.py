"""Chunk dispatch ownership using BF16 or FP8 payloads over stock MoonEP.

Combine continues to use the caller's full-width BF16 buffer. FP8 dispatch
borrows a half-width view of the same bytes and sends row descales explicitly.
No BF16 activation side payload or full reconstructed activation is allocated.
Forward owns its saved input when state is requested; output-only calls borrow
it through gate/up on the compute stream. Backward borrows its slot until the last dY reader;
input-gradient GEMM overwrites that slot only afterwards on the compute stream.
"""

import torch
import triton
from moonep.dispatch_epilogue import launch_dispatch_epilogue
from moonep.inter_rank_sync import launch_inter_rank_sync

from .activation_transport import FP8DispatchView, FP8Rows, _scatter_scale
from .kernels.quantize_rows import quantize_rows_fp8


class ChunkTransport:
    def __init__(self, buffers, precision, *, borrow_backward=True):
        self.borrow_backward = borrow_backward
        self.quantized = precision == "fp8"
        self.views = {
            id(buffer): FP8DispatchView(buffer) if self.quantized else buffer
            for buffer in buffers
        }

    def context(self, buffer):
        return self.views[id(buffer)]._require_ctx()

    def prepare(self, buffer, value, probabilities, *, reuse_plan):
        if not self.quantized:
            return value, probabilities, None
        payload, scales = quantize_rows_fp8(value)
        if reuse_plan:
            probabilities = (
                scales[:, None].expand(-1, self.context(buffer)["K"]).contiguous()
            )
        return payload.view(torch.bfloat16), probabilities, scales

    def collect(self, buffer, plan, scales, *, reuse_plan, retain_forward=True):
        ctx = self.context(buffer)
        launch_dispatch_epilogue(ctx, plan, pdl_launch=False)
        received = ctx["hidden_buf_local"]
        if (not reuse_plan and retain_forward) or (
            reuse_plan and not self.borrow_backward
        ):
            received = received.clone()
        metadata = (
            ctx["weights_buf_local"].view(torch.float32).clone()
            if self.quantized or not reuse_plan
            else None
        )
        if not self.quantized:
            return received, None if reuse_plan else metadata
        received = received.view(torch.float8_e4m3fn)
        if reuse_plan:
            return FP8Rows(received, metadata), None
        # Every rank first owns its routing probabilities before the metadata
        # channel is reused for source-token descales, matching main's recipe.
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
        received_scales = ctx["weights_buf_local"].view(torch.float32).clone()
        return FP8Rows(received, received_scales), metadata
