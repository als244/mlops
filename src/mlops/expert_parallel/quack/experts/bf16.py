"""SonicMoE Algorithm 2/3 expert math using QuACK's GEMM epilogues.

Inputs are already dispatched by MoonEP. No SonicMoE expert layer is invoked.
The first prototype supports BF16 compute and FP32 weight-gradient accumulation.
"""

from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.ops import ColVecLoad
from quack.gemm_interface import gemm, gemm_act, gemm_dact, gemm_tuned

from mlops.expert_parallel.quack.experts.policy import select_policy


@gemm_epilogue(ops={"score": ColVecLoad("score")})
def weighted_output(acc, score):
    return {"D": acc * score}


def _gemm(a, b, *, out, config, tuned, **kwargs):
    # Quack's ordinary wrapper has no explicit config argument.
    if config is None:
        return gemm(a, b, out=out, tuned=tuned, **kwargs)
    gemm_tuned.fn(a, b, out, config=config, **kwargs)
    return out


class QuackExperts:
    def __init__(self, *, tuned=False, model_config=None):
        self.tuned = tuned
        self.policy = select_policy(model_config, precision="bf16", tuned=tuned)

    def up(self, x, weight, cu):
        # Packed W1 has adjacent gate/up rows, established once at initialization.
        return gemm_act(
            x,
            weight.transpose(-1, -2),
            activation="swiglu",
            cu_seqlens_m=cu,
            store_preact=True,
            tuned=self.tuned,
            config=self.policy.up,
        )

    def down(self, activation, weight, probabilities, cu, out):
        weighted_output(
            activation,
            weight.transpose(-1, -2),
            out={"D": out},
            score=probabilities,
            config=self.policy.down,
            cu_seqlens_m=cu,
            tuned=self.tuned,
            dynamic_scheduler=False,
        )
        return out

    def down_backward(self, dy, weight, preact, probabilities, cu):
        # r = dy @ W2; dp = sum(r * swiglu(preact)); dpreact uses p*r.
        # The auxiliary output is p*swiglu(preact), used for W2's gradient.
        return gemm_dact(
            dy,
            weight,
            PreAct=preact,
            activation="swiglu",
            colvec_scale=probabilities,
            colvec_reduce=True,
            config=self.policy.down_backward,
            cu_seqlens_m=cu,
            tuned=self.tuned,
            dynamic_scheduler=False,
        )

    def input_gradient(self, dpreact, weight, cu, out):
        return _gemm(
            dpreact,
            weight,
            out=out,
            cu_seqlens_m=cu,
            tuned=self.tuned,
            config=self.policy.input_gradient,
        )

    def weight_gradient(self, x, dy, cu, bank):
        # Keep the existing two-bank ownership contract: PyTorch-owned returned
        # home gradients plus MoonEP-owned replica scratch. Two varlen-K calls
        # avoid a third gradient allocation and a full copy/split of the result.
        config = self.policy.weight_gradient(x.shape[-1], dy.shape[-1])
        q = bank.cfg.local_experts
        for offsets, destination in (
            (cu[: q + 1], bank.local_home_grad),
            (cu[q:], bank.local_replica_grad),
        ):
            # QuACK's FFI requires 16-byte alignment for the small offset
            # vector. A replica slice with q % 4 != 0 needs its own copy;
            # logical expert/token counts and activation storage are unchanged.
            if offsets.data_ptr() % 16:
                offsets = offsets.clone()
            _gemm(
                dy.T,
                x,
                out=destination[:, : bank.out_features, :],
                cu_seqlens_k=offsets,
                tuned=self.tuned,
                config=config,
            )
