"""Quack GEMMs for frozen projections and BF16 per-expert low-rank branches."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_act, gemm_tuned

from ..activation_transport import FP8Rows, quantized_rows
from ..experts.bf16 import _gemm
from ..experts.policy import select_policy
from ..experts.wgrad import scaled_fp8
from ..kernels.pointwise import _Pointwise

LORA_GEMM = GemmConfig(
    tile_m=128,
    tile_n=64,
    pingpong=False,
    cluster_m=1,
    is_dynamic_persistent=False,
    device_capacity=9,
)


def _dequantize(data, scales):
    return (data.float() * scales[:, None]).to(torch.bfloat16)


def _down_derivative(preact, r, probabilities):
    gate, up = preact[:, 0::2].float(), preact[:, 1::2].float()
    s = gate.sigmoid()
    activation = F.silu(gate) * up
    dh = r.float() * probabilities[:, None]
    dgate = dh * up * s * (1 + gate * (1 - s))
    dup = dh * F.silu(gate)
    dpreact = torch.stack((dgate, dup), dim=-1).flatten(1).to(torch.bfloat16)
    return (
        dpreact,
        (activation * probabilities[:, None]).to(torch.bfloat16),
        (r.float() * activation).sum(-1),
    )


def _scale(value, probabilities, out):
    out.copy_((value.float() * probabilities[:, None]).to(value.dtype))
    return out


@lru_cache(None)
def compiled(fn):
    return torch.compile(
        fn,
        fullgraph=True,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )


def bf16_rows(value):
    return (
        compiled(_dequantize)(value.data, value.scales)
        if isinstance(value, FP8Rows)
        else value
    )


class LoRAExperts:
    needs_host_offsets = False  # Every trainable wgrad uses BF16 grouped varlen-K.

    def __init__(self, config, lora, banks):
        self.cfg, self.lora, self.banks = config, lora, banks
        self.policy = select_policy(
            config, precision=config.compute_precision, tuned=config.gemm_tuned
        )
        self.pw = _Pointwise()

    def _linear(self, x, weight, cu, *, dgrad=False, out=None, config=None):
        if self.cfg.compute_precision == "bf16":
            shape = (x.shape[0], weight.shape[-1 if dgrad else -2])
            if out is None:
                out = x.new_empty(shape)
            _gemm(
                x,
                weight if dgrad else weight.mT,
                out=out,
                config=config,
                cu_seqlens_m=cu,
                tuned=self.cfg.gemm_tuned,
            )
        else:
            qx, sx = quantized_rows(x)
            qw, qt, sw, st = weight
            matrix, scales = (qt, st) if dgrad else (qw, sw)
            if out is None:
                out = torch.empty(
                    (qx.shape[0], matrix.shape[-2]),
                    device=qx.device,
                    dtype=torch.bfloat16,
                )
            scaled_fp8(
                qx,
                matrix.mT,
                out={"D": out},
                xs=sx,
                ws=scales,
                cu_seqlens_m=cu,
                config=config,
                tuned=self.cfg.gemm_tuned,
            )
        self.pw.mask_tail(out, cu[-1:])
        return out

    def _low(self, x, weight, cu, *, transpose=True, out=None, add=None, scale=1.0):
        x = bf16_rows(x)
        if out is None:
            out = x.new_empty((x.shape[0], weight.shape[-2 if transpose else -1]))
        gemm_tuned.fn(
            x,
            weight.mT if transpose else weight,
            out,
            C=add,
            alpha=float(scale),
            config=LORA_GEMM,
            cu_seqlens_m=cu,
        )
        self.pw.mask_tail(out, cu[-1:])
        return out

    def up(self, x, weight, cu):
        base = self._linear(x, weight, cu, config=self.policy.up)
        a, b = self.banks[0].factors.compute_weights
        z = self._low(x, a, cu)
        preact, activation = gemm_act(
            z,
            b.mT,
            C=base,
            alpha=float(self.lora.scale),
            activation="swiglu",
            cu_seqlens_m=cu,
            config=LORA_GEMM,
            tuned=False,
        )
        self.pw.mask_tail(preact, cu[-1:])
        self.pw.mask_tail(activation, cu[-1:])
        return preact, activation

    def down(self, activation, weight, probabilities, cu, out):
        base = self._linear(activation, weight, cu, config=self.policy.down)
        a, b = self.banks[1].factors.compute_weights
        z = self._low(activation, a, cu)
        value = self._low(z, b, cu, add=base, scale=self.lora.scale)
        return compiled(_scale)(value, probabilities, out)

    def down_backward(self, dy, weight, preact, probabilities, cu):
        r = self._linear(dy, weight, cu, dgrad=True, config=self.policy.down_backward)
        a, b = self.banks[1].factors.compute_weights
        dz = self._low(dy, b, cu, transpose=False, scale=self.lora.scale)
        r = self._low(dz, a, cu, transpose=False, add=r)
        return compiled(_down_derivative)(preact, r, probabilities)

    def input_gradient(self, dpreact, weight, cu, out):
        base = self._linear(
            dpreact, weight, cu, dgrad=True, out=out, config=self.policy.input_gradient
        )
        a, b = self.banks[0].factors.compute_weights
        dz = self._low(dpreact, b, cu, transpose=False, scale=self.lora.scale)
        return self._low(dz, a, cu, transpose=False, add=base, out=out)

    def weight_gradient(self, x, dy, cu, bank, *, accumulate_home=False, offsets=None):
        x, dy = bf16_rows(x), bf16_rows(dy)
        a, b = bank.factors.compute_weights
        z = self._low(x, a, cu)
        dz = self._low(dy, b, cu, transpose=False, scale=self.lora.scale)
        q = self.cfg.local_experts
        for index, bounds in enumerate((cu[: q + 1], cu[q:])):
            if bounds.data_ptr() % 16:
                bounds = bounds.clone()
            da, db = bank.factors.grad_views[index]
            for left, right, out, alpha in (
                (dz.T, x, da, 1.0),
                (dy.T, z, db, self.lora.scale),
            ):
                gemm_tuned.fn(
                    left,
                    right,
                    out,
                    alpha=float(alpha),
                    config=LORA_GEMM,
                    cu_seqlens_k=bounds,
                    add_to_output=accumulate_home and index == 0,
                )
