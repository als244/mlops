"""Base matrix products, with explicit storage and GEMM recipes."""

import torch

from . import fp8, nvfp4
from .storage import DENSE, WeightSpec, materialize, split_components


def add_product(out, left, right, scale=1.0):
    if out.dtype == left.dtype:
        torch.addmm(out, left, right, alpha=scale, out=out)
    elif out.dtype == torch.float32:
        torch.ops.aten.addmm.dtype_out(
            out, left, right, out.dtype, alpha=scale, out=out
        )
    else:
        # A different returned-gradient dtype must not change GEMM operands.
        product = torch.empty(out.shape, device=out.device, dtype=torch.float32)
        torch.ops.aten.mm.dtype_out(left, right, torch.float32, out=product)
        torch.add(out, product, alpha=scale, out=out)


class BaseGEMMs:
    def __init__(
        self,
        weights,
        precision,
        specs,
        x,
        hidden,
        capacity,
        *,
        backward=False,
        train_base=False,
    ):
        self.precision = precision
        specs = [WeightSpec.parse(s) for s in specs.split("|")]
        parts = split_components(weights, specs)
        widths = {x.shape[1], hidden}
        if backward:
            widths.add(2 * hidden)
        self.dense = precision in DENSE
        if self.dense:
            self.dtype = DENSE[precision]
            self.weights = [
                materialize(v, s, self.dtype) for v, s in zip(parts, specs, strict=True)
            ]
            # Fixed-capacity conversion buffers when activation and GEMM dtype differ.
            self.inputs = (
                {
                    w: torch.empty(capacity, w, device=x.device, dtype=self.dtype)
                    for w in widths
                }
                if self.dtype != x.dtype
                else {}
            )
            self.outputs = (
                {
                    w: torch.empty(capacity, w, device=x.device, dtype=self.dtype)
                    for w in {x.shape[1], hidden, 2 * hidden}
                }
                if self.inputs
                else {}
            )
            return
        self.forward_weights, self.backward_weights = [], []
        for values in parts:
            data, scales, *extra = values
            if precision == "fp8":
                self.forward_weights.append((data, scales))
                if backward:
                    self.backward_weights.append(
                        (data.T.contiguous(), scales.T.contiguous())
                    )
            else:
                global_scale = extra[0]
                m, k = data.shape[0], data.shape[1] * 2
                self.forward_weights.append(
                    (data, nvfp4.weight_scales(scales, m, k), global_scale)
                )
                if backward:
                    self.backward_weights.append(
                        (
                            nvfp4.transpose_packed(data),
                            nvfp4.weight_scales(scales, k, m, transpose=True),
                            global_scale,
                        )
                    )
        quantizer = fp8.Quantizer if precision == "fp8" else nvfp4.Quantizer
        self.quantizers = {w: quantizer(capacity, w, x.device) for w in widths}
        if train_base and precision == "fp8":
            self.wgrad_left = {
                w: fp8.Quantizer(w, capacity, x.device)
                for w in {x.shape[1], 2 * hidden}
            }
            self.wgrad_right = {
                w: fp8.Quantizer(w, capacity, x.device) for w in {x.shape[1], hidden}
            }

    def _dense_mm(self, x, weight, out):
        if not self.inputs:
            torch.mm(x, weight, out=out)
            return
        operand = self.inputs[x.shape[1]][: x.shape[0]]
        result = self.outputs[out.shape[1]][: out.shape[0]]
        operand.copy_(x)
        torch.mm(operand, weight, out=result)
        out.copy_(result)

    def forward(self, index, x, out):
        if self.dense:
            self._dense_mm(x, self.weights[index].T, out)
        else:
            backend = fp8 if self.precision == "fp8" else nvfp4
            backend.gemm(
                self.quantizers[x.shape[1]](x), self.forward_weights[index], out
            )

    def backward(self, index, dy, out):
        if self.dense:
            self._dense_mm(dy, self.weights[index], out)
        else:
            backend = fp8 if self.precision == "fp8" else nvfp4
            backend.gemm(
                self.quantizers[dy.shape[1]](dy), self.backward_weights[index], out
            )

    def weight_gradient(self, index, dy, x, out):
        if self.precision == "fp8":
            qdy = self.wgrad_left[dy.shape[1]](dy.T)
            qx = self.wgrad_right[x.shape[1]](x.T)
            fp8.gemm(qdy, qx, out, weight_blocks=False, add=True)
        else:
            # NVFP4 uses the documented conservative BF16/FP16 weight-gradient
            # recipe. Dense products use the explicitly requested GEMM dtype.
            if self.dense and self.inputs:
                # Distinct fixed-capacity banks remain safe when widths coincide.
                lhs = self.outputs[dy.shape[1]][: dy.shape[0]]
                rhs = self.inputs[x.shape[1]][: x.shape[0]]
                lhs.copy_(dy)
                rhs.copy_(x)
                dy, x = lhs, rhs
            add_product(out, dy.T, x)
