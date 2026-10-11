"""Independent physical storage, activation, GEMM, and gradient settings."""

from dataclasses import dataclass

import torch

from .precision import validate_config


@dataclass(frozen=True)
class SequentialMoEConfig:
    hidden_size: int
    expert_size: int
    num_experts: int
    top_k: int
    rank: int = 32
    alpha: float = 32.0
    chunk_capacity: int = 8192
    row_multiple: int = 256
    weight_storage: str = "bf16"
    gemm_precision: str = "bf16"
    nvfp4_block_rows: int = 16
    activation_dtype: torch.dtype = torch.bfloat16
    init_dtype: torch.dtype | None = None
    initializer_range: float | None = None
    train_router: bool = True
    train_shared: bool = True
    factor_dtype: torch.dtype = torch.bfloat16
    gradient_dtype: torch.dtype = torch.float32
    router_dtype: torch.dtype = torch.float32
    activation_accum_dtype: torch.dtype | None = None
    shared_experts: int = 0
    shared_expert_size: int | None = None
    swiglu_limit: float | None = None

    def __post_init__(self):
        # Same-dtype transitions avoid a full [T,D] conversion at the layer end.
        # An explicit dtype (including FP32) remains available for accumulation.
        if self.activation_accum_dtype is None:
            object.__setattr__(self, "activation_accum_dtype", self.activation_dtype)
        for name in (
            "hidden_size",
            "expert_size",
            "num_experts",
            "rank",
            "chunk_capacity",
            "row_multiple",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if self.chunk_capacity % self.row_multiple:
            raise ValueError("chunk_capacity must be a multiple of row_multiple")
        if self.shared_experts < 0 or (
            self.shared_expert_size is not None and self.shared_expert_size <= 0
        ):
            raise ValueError("Invalid shared expert shape")
        # 'dense' is accepted for recipe-based test/config generation.
        dense = (
            str(self.activation_dtype)
            .removeprefix("torch.")
            .replace("bfloat16", "bf16")
            .replace("float16", "fp16")
        )
        for name in ("weight_storage", "gemm_precision"):
            if getattr(self, name) == "dense":
                object.__setattr__(self, name, dense)
        if self.swiglu_limit is not None and self.swiglu_limit <= 0:
            raise ValueError("swiglu_limit must be positive or None")
        if self.initializer_range is not None and self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive or None")
        validate_config(self)

    @property
    def scale(self):
        return self.alpha / self.rank

    @property
    def storage_kind(self):
        return (
            "dense"
            if self.weight_storage in ("bf16", "fp16", "fp32")
            else self.weight_storage
        )

    def rows_for(self, valid):
        return (valid + self.row_multiple - 1) // self.row_multiple * self.row_multiple
