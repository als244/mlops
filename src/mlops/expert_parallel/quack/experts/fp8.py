"""Experimental SM90 FP8 expert math over MoonEP's planned group rows.

Group lengths include zero padding when the layer enables token padding.

Forward and activation gradients use grouped GEMMs with scale-aware fused
epilogues. Weight gradients currently copy offsets to the host and invoke one
GEMM per expert. This measurable limitation is reported by the layer, not hidden
behind a BF16 fallback.
"""

import itertools

import torch
from quack.activation import dswiglu, swiglu
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.math import pack, unpack
from quack.epilogue.ops import ColVecLoad, ColVecReduce, RowVecLoad

from mlops.expert_parallel.quack.activation_transport import (
    gradient_rows,
    quantized_rows,
)
from mlops.expert_parallel.quack.experts.policy import select_policy
from mlops.expert_parallel.quack.experts.wgrad import expert_weight_gradient, scaled_fp8
from mlops.expert_parallel.quack.kernels.quantize_rows import (
    quantize_rows_fp8 as quantize_act_per_token_fp8,
)


@gemm_epilogue(
    outputs=("postact",),
    mode="acc_pair",
    ops={"xs": ColVecLoad("xs"), "ws": RowVecLoad("ws")},
)
def scaled_swiglu(acc, xs, ws):
    value = acc * xs * ws
    gate, up = unpack(value)
    return {"D": value, "postact": swiglu(gate, up)}


@gemm_epilogue(
    ops={"xs": ColVecLoad("xs"), "ws": RowVecLoad("ws"), "score": ColVecLoad("score")}
)
def scaled_weighted_output(acc, xs, ws, score):
    return {"D": acc * xs * ws * score}


@gemm_epilogue(
    outputs=("postact",),
    mode="packed_cd_b16x2",
    ops={"xs": ColVecLoad("xs"), "ws": RowVecLoad("ws"), "score": ColVecLoad("score")},
    reduces={"dscore": ColVecReduce("dscore", scaled=True)},
)
def scaled_swiglu_backward(acc, c, xs, ws, score):
    gate, up = unpack(c)
    r = acc * xs * ws
    dg, du, activation = dswiglu(gate, up, r * score)
    return {"D": pack(dg, du), "postact": activation * score, "dscore": (activation, r)}


class QuackFP8Experts:
    def __init__(self, *, tuned=False, model_config=None):
        self.tuned = tuned
        self.policy = select_policy(model_config, precision="fp8_current", tuned=tuned)

    def up(self, x, weight, cu):
        qx, sx = quantized_rows(x)
        qw, _, sw, _ = weight
        result = scaled_swiglu(
            qx,
            qw.transpose(-1, -2),
            xs=sx,
            ws=sw,
            out_dtype=torch.bfloat16,
            cu_seqlens_m=cu,
            tuned=self.tuned,
            config=self.policy.up,
        )
        return result["D"], result["postact"]

    def down(self, activation, weight, probabilities, cu, out):
        qa, sa = quantize_act_per_token_fp8(activation)
        qw, _, sw, _ = weight
        scaled_weighted_output(
            qa,
            qw.transpose(-1, -2),
            out={"D": out},
            xs=sa,
            ws=sw,
            score=probabilities,
            cu_seqlens_m=cu,
            tuned=self.tuned,
            config=self.policy.down,
        )
        return out

    def down_backward(self, dy, weight, preact, probabilities, cu):
        qdy, sdy = quantized_rows(dy)
        _, qt, _, st = weight
        result = scaled_swiglu_backward(
            qdy,
            qt.transpose(-1, -2),
            C=preact,
            xs=sdy,
            ws=st,
            score=probabilities,
            cu_seqlens_m=cu,
            out_dtype=torch.bfloat16,
            tuned=self.tuned,
            config=self.policy.down_backward,
        )
        return result["D"], result["postact"], result["dscore"]

    def input_gradient(self, dpreact, weight, cu, out):
        qd, sd = quantize_act_per_token_fp8(dpreact)
        _, qt, _, st = weight
        scaled_fp8(
            qd,
            qt.transpose(-1, -2),
            out={"D": out},
            xs=sd,
            ws=st,
            cu_seqlens_m=cu,
            tuned=self.tuned,
            config=self.policy.input_gradient,
        )
        return out

    def weight_gradient(self, x, dy, cu, bank):
        # SM90 varlen-K currently disallows the layouts required by FP8 WGMMA.
        # Keep this host synchronization visible in NVTX/profiles until a device
        # grouped scheduler supports the packed transposed operands directly.
        x, x_scales = gradient_rows(x)
        dy, dy_scales = gradient_rows(dy)
        config = self.policy.weight_gradient(x.shape[-1], dy.shape[-1])
        with torch.cuda.nvtx.range("moon_quack/fp8_wgrad/offsets_to_host"):
            offsets = cu.cpu().tolist()
        q = bank.cfg.local_experts
        for expert, (start, end) in enumerate(itertools.pairwise(offsets)):
            if (
                expert >= q
                and start == end
                and getattr(bank, "replica_grad_cleared_after_reduce", False)
            ):
                continue
            destination = (
                bank.local_home_grad if expert < q else bank.local_replica_grad
            )[expert % q]
            with torch.cuda.nvtx.range("moon_quack/fp8_wgrad/expert"):
                expert_weight_gradient(
                    x[start:end],
                    dy[start:end],
                    out=destination[: bank.out_features],
                    x_scales=None if x_scales is None else x_scales[start:end],
                    dy_scales=None if dy_scales is None else dy_scales[start:end],
                    tuned=self.tuned,
                    config=config,
                )
