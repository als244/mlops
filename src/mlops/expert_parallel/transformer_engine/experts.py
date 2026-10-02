"""Transformer Engine grouped GEMMs, recipes and capability checks."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .config import MoEConfig
from .kernels.pointwise import _Pointwise


def element_offsets(counts: Tensor, width: int) -> Tensor:
    return torch.cat((torch.zeros_like(counts[:1]), counts.cumsum(0))) * width


def validate_te_hardware_precision(c: MoEConfig) -> dict[str, Any]:
    """Fail closed on unsupported TE grouped-tensor hardware/precision combinations.

    Hopper support deliberately requires the current grouped-tensor cuBLASLt path:
    SM90 and cuBLASLt >= 13.4.  Blackwell requires >= 13.3.  MXFP8/NVFP4
    grouped-tensor execution is Blackwell-only.  This validates eligibility; the
    quantized expert backend additionally performs its own TE availability check.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cc = torch.cuda.get_device_capability()
    try:
        import transformer_engine.pytorch  # noqa: F401 -- Registers Transformer Engine's C++/CUDA extension module.
        import transformer_engine_torch as tex

        cublaslt = int(tex.get_cublasLt_version())
    except Exception as e:
        raise RuntimeError(
            "Transformer Engine with get_cublasLt_version() is required"
        ) from e
    if cc < (9, 0):
        raise RuntimeError(
            f"MoonEP+TE grouped-tensor backend requires Hopper SM90+; got SM{cc[0]}{cc[1]}"
        )
    minimum = {"bf16": 130400, "fp8_current": 130500, "fp8_block": 130600}[
        c.compute_precision
    ]
    if cc < (10, 0) and cublaslt < minimum:
        raise RuntimeError(
            f"Hopper {c.compute_precision} grouped GEMM requires cuBLASLt >= {minimum}; got {cublaslt}"
        )
    if cc >= (10, 0) and cublaslt < 130300:
        raise RuntimeError(
            f"Blackwell grouped-tensor GEMM requires cuBLASLt >= 13.3; got {cublaslt}"
        )
    if c.compute_precision in ("mxfp8", "nvfp4") and cc < (10, 0):
        raise RuntimeError(
            f"{c.compute_precision} grouped-tensor expert GEMM requires Blackwell SM100+"
        )
    # Transformer Engine block, current and delayed FP8 formats are supported on Hopper.
    return {
        "compute_capability": cc,
        "cublaslt_version": cublaslt,
        "precision": c.compute_precision,
    }


def make_te_recipe(name: str):
    """Return the TE recipe for a requested compute precision."""
    if name == "bf16":
        return None
    from transformer_engine.common import recipe

    if name == "fp8_current":
        return recipe.Float8CurrentScaling(fp8_format=recipe.Format.E4M3)
    if name == "fp8_delayed":
        return recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
    if name == "fp8_block":
        # Transformer Engine block-scaled FP8 for Hopper.
        cls = getattr(recipe, "Float8BlockScaling", None)
        if cls is None:
            raise RuntimeError("Installed TE does not expose Float8BlockScaling")
        return cls(w_block_scaling_dim=1)
    if name == "mxfp8":
        return recipe.MXFP8BlockScaling()
    if name == "nvfp4":
        return recipe.NVFP4BlockScaling()
    raise ValueError(name)


class _TEBackend:
    def __init__(self, cfg: MoEConfig):
        self.hw = validate_te_hardware_precision(cfg)
        if cfg.compute_precision == "fp8_block" and 2 * cfg.local_experts > 64:
            raise NotImplementedError(
                "FP8 block scaling currently supports at most 64 home/replica groups; "
                "larger grouped slices require rebased block-scale metadata. "
                "BF16 and FP8 current scaling support chunked groups."
            )
        self.recipe = make_te_recipe(cfg.compute_precision)
        self.cfg = cfg
        try:
            from transformer_engine.pytorch.cpp_extensions.gemm import (
                general_grouped_gemm_for_grouped_tensor,
            )
            from transformer_engine.pytorch.ops.basic.grouped_linear import (
                is_op_fuser_grouped_tensor_path_supported,
            )
            from transformer_engine.pytorch.tensor import GroupedTensorStorage
        except ImportError as e:
            raise RuntimeError(
                "Required TE grouped-tensor APIs are absent; no alternative GEMM fallback is supplied"
            ) from e
        if not is_op_fuser_grouped_tensor_path_supported(self.recipe, torch.bfloat16):
            raise RuntimeError(
                f"The installed TE/GPU/cuBLASLt combination does not support {cfg.compute_precision} grouped tensors"
            )
        self.storage_type = GroupedTensorStorage
        self.gemm = general_grouped_gemm_for_grouped_tensor
        self.quantizer = (
            None
            if cfg.compute_precision == "bf16"
            else _make_quantizer(cfg.compute_precision)
        )

    def storage(self, x, counts, offsets=None):
        return self.storage_type(
            shape=tuple(x.shape),
            dtype=x.dtype,
            num_tensors=counts.numel(),
            quantizer=None,
            data=x.reshape(-1),
            first_dims=counts,
            tensor_offsets=element_offsets(counts, x.shape[1])
            if offsets is None
            else offsets,
        )

    def group_chunks(self, counts, *tensors):
        # The installed TE discrete-input/output API has a 64-pointer kernel
        # argument limit. Slice GPU metadata, retaining absolute payload offsets;
        # no token readback, packing copy or expert-count specialization is needed.
        if counts.numel() <= 64:
            yield slice(0, counts.numel()), counts, [None] * len(tensors)
            return
        offsets = [element_offsets(counts, x.shape[1]) for x in tensors]
        for start in range(0, counts.numel(), 64):
            end = min(start + 64, counts.numel())
            yield (
                slice(start, end),
                counts[start:end],
                [v[start : end + 1] for v in offsets],
            )

    def linear(
        self, x, weights, counts, dgrad=False, *, out=None, accumulate=False, alpha=None
    ):
        if accumulate and out is None:
            raise ValueError("GEMM accumulation requires an initialized output")
        if out is None:
            out = torch.empty(
                (x.shape[0], weights[0].shape[1 if dgrad else 0]),
                device=x.device,
                dtype=x.dtype,
            )
        for groups, chunk_counts, (xoff, yoff) in self.group_chunks(counts, x, out):
            self.gemm(
                weights[groups],
                self.operand(x, chunk_counts, xoff),
                self.storage(out, chunk_counts, yoff),
                layout="NN" if dgrad else "TN",
                use_split_accumulator=dgrad and self.quantizer is not None,
                accumulate=accumulate,
                alpha=alpha,
            )
        self.clear_output_tail(out, counts)
        return out

    def clear_output_tail(self, out, counts):
        # beta=0 overwrites every grouped row; only the unused capacity is
        # unwritten. Keep it zero so full-capacity pointwise math cannot see NaNs.
        _Pointwise().mask_tail(out, counts.sum().reshape(1))

    def wgrad(self, x, dy, out, counts, *, alpha=None):
        for groups, chunk_counts, (xoff, yoff) in self.group_chunks(counts, x, dy):
            self.gemm(
                self.operand(x, chunk_counts, xoff),
                self.operand(dy, chunk_counts, yoff),
                out[groups],
                layout="NT",
                accumulate=False,
                use_split_accumulator=self.quantizer is not None,
                alpha=alpha,
            )

    def operand(self, x, counts, offsets=None):
        if self.quantizer is None:
            return self.storage(x, counts, offsets)
        import transformer_engine_torch as tex

        return tex.group_quantize(
            x, self.quantizer, counts.numel(), counts, tensor_offsets=offsets
        )


def _make_quantizer(precision):
    import transformer_engine_torch as tex
    from transformer_engine.pytorch import (
        Float8BlockQuantizer,
        Float8CurrentScalingQuantizer,
    )

    if precision == "fp8_current":
        return Float8CurrentScalingQuantizer(
            tex.DType.kFloat8E4M3, device="cuda", rowwise=True, columnwise=True
        )
    if precision == "fp8_block":
        return Float8BlockQuantizer(
            tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=1,
            force_pow_2_scales=True,
        )
    raise ValueError(precision)
