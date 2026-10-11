"""Validation of supported precision combinations before model allocation."""

import torch

STORAGE = {"bf16", "fp16", "fp32", "fp8", "nvfp4", "mxfp4", "int4"}
GEMM = {"bf16", "fp16", "fp8", "nvfp4"}
FLOAT = {torch.bfloat16, torch.float16, torch.float32}


def validate_config(config):
    c = config
    if c.weight_storage not in STORAGE:
        raise ValueError(
            f"Unsupported weight_storage={c.weight_storage!r}; choose {sorted(STORAGE)}"
        )
    if c.gemm_precision not in GEMM:
        raise ValueError(
            f"Unsupported gemm_precision={c.gemm_precision!r}; choose {sorted(GEMM)}"
        )
    if c.activation_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("activation_dtype must be torch.bfloat16 or torch.float16")
    for name in (
        "gradient_dtype",
        "factor_dtype",
        "router_dtype",
        "activation_accum_dtype",
    ):
        if getattr(c, name) not in FLOAT:
            raise ValueError(f"{name} must be FP32, BF16, or FP16")
    if c.init_dtype is not None and c.init_dtype not in FLOAT:
        raise ValueError("init_dtype must be FP32, BF16, FP16, or None")
    if c.nvfp4_block_rows not in (1, 16):
        raise ValueError("nvfp4_block_rows must be 1 or 16")
    if c.gemm_precision == "nvfp4" and c.nvfp4_block_rows != 16:
        raise ValueError(
            "NVFP4 GEMMs require 16x16 weight scales; use BF16/FP16 GEMMs for 1x16"
        )
    # Explicitly supported quantized GEMM recipes. Storage-only quantization works
    # with either BF16 or FP16 GEMMs for every supported checkpoint format.
    if c.gemm_precision in ("fp8", "nvfp4") and c.weight_storage != c.gemm_precision:
        raise ValueError(
            f"Unsupported weight_storage={c.weight_storage!r} with "
            f"gemm_precision={c.gemm_precision!r}; use BF16/FP16 GEMMs, or matching "
            "FP8/NVFP4 storage. Cross-format requantization is not implicit."
        )
    if c.weight_storage == "nvfp4" and (c.hidden_size % 32 or c.expert_size % 32):
        raise ValueError(
            "NVFP4 initialization needs hidden_size/expert_size divisible by 32"
        )
    if c.gemm_precision == "nvfp4" and c.row_multiple % 128:
        raise ValueError("NVFP4 GEMMs need row_multiple divisible by 128")


def validate_device(config, device):
    device = torch.device(device)
    if device.type == "meta":
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("SequentialMoE currently requires a CUDA device")
    major, minor = torch.cuda.get_device_capability(device)
    if (
        config.activation_dtype == torch.bfloat16
        or config.gemm_precision == "bf16"
        or config.router_dtype == torch.bfloat16
    ) and major < 8:
        raise ValueError(
            "BF16 activation/GEMM/router configuration requires SM80 or newer"
        )
    if config.gemm_precision == "fp8" and (major, minor) < (8, 9):
        raise ValueError(
            "FP8 GEMMs require SM89 or newer; BF16/FP16 GEMMs can decode FP8 storage"
        )
    if config.gemm_precision == "nvfp4" and major < 10:
        raise ValueError("NVFP4 GEMMs require Blackwell (SM100/SM120 family) or newer")
    if config.gemm_precision == "nvfp4" and not hasattr(
        torch.ops.aten, "_scaled_mm_v2"
    ):
        raise ValueError("NVFP4 GEMMs require PyTorch with aten._scaled_mm_v2")


def validate_weights(config, specs, components):
    """Validate a complete expert before replacing any of its current parameters."""
    expected = (
        (2 * config.expert_size, config.hidden_size),
        (config.hidden_size, config.expert_size),
    )
    for spec, parts, shape in zip(specs, components, expected, strict=True):
        spec.validate(parts)
        if (spec.rows, spec.cols) != shape:
            raise ValueError(
                f"Checkpoint weight shape {(spec.rows, spec.cols)} does not match {shape}"
            )
        if spec.format != config.weight_storage:
            raise ValueError(
                f"Checkpoint has {spec.format} weights; config requests {config.weight_storage}"
            )
        if config.gemm_precision == "fp8" and parts[1].dtype != torch.float32:
            raise ValueError(
                "FP8 GEMMs require FP32 weight_scale_inv dequantization multipliers"
            )
        if config.gemm_precision == "fp8" and (spec.block_rows, spec.block_cols) != (
            128,
            128,
        ):
            raise ValueError(
                "FP8 GEMMs currently require 128x128 weight blocks; "
                "use BF16/FP16 GEMMs for per-channel/per-tensor FP8 checkpoints"
            )
        if config.gemm_precision == "nvfp4" and (
            (spec.block_rows, spec.block_cols) != (16, 16) or spec.global_rows
        ):
            raise ValueError(
                "NVFP4 GEMMs currently require the prototype's 16x16 scale blocks. "
                "Checkpoint 1x16 scales support BF16/FP16 GEMMs; a lossless "
                "NVFP4 backward transpose is not available for that layout."
            )
