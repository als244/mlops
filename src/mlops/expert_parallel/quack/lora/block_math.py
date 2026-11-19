"""EP LoRA using the existing single-copy Hopper block-FP8 projections.

The frozen base stays in gate-then-up checkpoint order. Factors and preactivation
rows retain Quack EP's adjacent gate/up order; the shared projection and input
gradient kernels perform that mapping without a transposed weight allocation.
Communication, factor gradients and accumulation use the existing LoRA runtime.
"""

from mlops.kernels.quack_moe.pointwise import _Pointwise
from mlops.providers.sonicmoe.block_fp8 import (
    BlockWeight,
    input_gradient_rows,
    lora_down_projection,
    lora_up_projection,
)

from .math import LoRAExperts, bf16_rows, down_derivative, scaled_rows


def _weight(components):
    # MoonEP aligns each expert's communication slot. The GEMM reads a compact
    # scale grid; gather only that small grid, never the FP8 weight payload.
    return BlockWeight(components[0], components[1].contiguous())


class BlockLoRAExperts(LoRAExperts):
    def __init__(self, config, lora, banks):
        self.cfg, self.lora, self.banks = config, lora, banks
        self.pw = _Pointwise()

    def up(self, x, weight, cu):
        x = bf16_rows(x)
        a, b = self.banks[0].factors.compute_weights
        z = self._low(x, a, cu)
        contribution = self._low(z, b, cu, scale=self.lora.scale)
        preact, activation = lora_up_projection(
            x, _weight(weight), contribution, cu, limit=self.cfg.swiglu_limit
        )
        self.pw.mask_tail(preact, cu[-1:])
        self.pw.mask_tail(activation, cu[-1:])
        return preact, activation

    def down(self, activation, weight, probabilities, cu, out):
        a, b = self.banks[1].factors.compute_weights
        z = self._low(activation, a, cu)
        contribution = self._low(z, b, cu, scale=self.lora.scale)
        value = lora_down_projection(activation, _weight(weight), contribution, cu)
        self.pw.mask_tail(value, cu[-1:])
        return scaled_rows(value, probabilities, out)

    def down_backward(self, dy, weight, preact, probabilities, cu):
        r = input_gradient_rows(bf16_rows(dy), _weight(weight), cu)
        a, b = self.banks[1].factors.compute_weights
        dz = self._low(dy, b, cu, transpose=False, scale=self.lora.scale)
        r = self._low(dz, a, cu, transpose=False, add=r)
        return down_derivative(preact, r, probabilities, self.cfg.swiglu_limit)

    def input_gradient(self, dpreact, weight, cu, out):
        base = input_gradient_rows(
            bf16_rows(dpreact), _weight(weight), cu, gate_up=True
        )
        a, b = self.banks[0].factors.compute_weights
        dz = self._low(dpreact, b, cu, transpose=False, scale=self.lora.scale)
        return self._low(dz, a, cu, transpose=False, add=base, out=out)
