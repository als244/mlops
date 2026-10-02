"""FP8 chunk execution using the existing Quack quantizers and GEMM epilogues.

Row scales for forward/dgrad keep their reduction dimension. Weight-gradient
feature scales reduce over each chunk's expert rows. This is a different FP8
quantization domain from the unchunked layer, and is reported explicitly.
"""

import itertools

import torch

from ..activation_transport import gradient_rows
from ..experts.fp8 import QuackFP8Experts
from ..experts.wgrad import scaled_fp8


class ChunkFP8Experts(QuackFP8Experts):
    def weight_gradient(self, x, dy, cu, bank, *, accumulate_home=False, offsets=None):
        x, x_scales = gradient_rows(x)
        dy, dy_scales = gradient_rows(dy)
        config = self.policy.weight_gradient(x.shape[-1], dy.shape[-1])
        if offsets is None:
            with torch.cuda.nvtx.range("moon_quack/fp8_wgrad/offsets_to_host"):
                offsets = cu.cpu().tolist()
        from .quantize import quantize_groups

        with torch.cuda.nvtx.range("moon_quack/fp8_wgrad/group_quantize"):
            qx_all, sx_all = quantize_groups(x, cu, offsets, row_scales=x_scales)
            qdy_all, sdy_all = quantize_groups(dy, cu, offsets, row_scales=dy_scales)
        q = bank.cfg.local_experts
        for expert, (start, end) in enumerate(itertools.pairwise(offsets)):
            add = accumulate_home and expert < q
            destination = (
                bank.local_home_grad if expert < q else bank.local_replica_grad
            )[expert % q, : bank.out_features]
            with torch.cuda.nvtx.range("moon_quack/fp8_wgrad/expert"):
                if start == end:
                    if expert < q and not add:
                        destination.zero_()
                    # Replica scratch was cleared by the previous reduction;
                    # an empty later chunk must not clear an accumulated home.
                    continue
                qdy = qdy_all[start * dy.shape[1] : end * dy.shape[1]].view(
                    dy.shape[1], end - start
                )
                qx = qx_all[start * x.shape[1] : end * x.shape[1]].view(
                    x.shape[1], end - start
                )
                sdy, sx = sdy_all[expert], sx_all[expert]
                scaled_fp8(
                    qdy,
                    qx.T,
                    out={"D": destination},
                    xs=sdy,
                    ws=sx,
                    tuned=self.tuned,
                    config=config,
                    add_to_output=add,
                )
