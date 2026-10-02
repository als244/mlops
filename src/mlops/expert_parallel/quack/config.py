"""QuACK configuration; direct transpose quantization is the only wgrad path."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class MoEConfig:
    ep_size: int
    num_experts: int
    top_k: int
    model_dim: int
    expert_hidden_dim: int
    tokens_per_rank: int | None = (
        None  # Legacy shape hint; new callers supply a buffer.
    )
    latent_dim: int | None = None
    num_shared_experts: int = 0
    shared_expert_dim: int | None = None
    token_padding: int = 128
    num_comm_sms: int = 32
    overlap: bool = True
    renormalize_topk: bool = True
    weight_grad_dtype: torch.dtype | None = (
        None  # FP32 default; FP32 GEMM/reduction scratch in either case.
    )
    parameter_dtype: torch.dtype | None = (
        None  # Compatibility alias for logical gradient dtype, not compute storage.
    )
    # FP8 expert parameters use explicit QuACK payloads and scales with ordinary FP32 gradients.
    # Any high-precision optimizer masters live outside this module.
    weight_transport: str | None = (
        None  # Derived from compute_precision; legacy explicit values are checked.
    )
    compute_precision: str = "bf16"  # bf16 | fp8_current
    activation_transport: str = "bf16"  # FP8 transport is an explicit opt-in.
    reuse_communication_buffers: bool = True
    # Layers using the same caller-owned token buffer can also reuse expert
    # publication/reduction storage. Their parameters remain independently owned.
    share_expert_banks: bool = False
    gradient_output_mode: str = (
        "owned"  # owned | copy; identical policy for every precision
    )
    fuse_probability_backward: bool = True
    fuse_input_grad_accumulation: bool = True
    router_dtype: torch.dtype = torch.float32
    router_weight_grad_dtype: torch.dtype = torch.float32
    init_std: float = 0.02
    profile_ranges: bool = True
    retain_intermediates: bool = False

    num_chunks: int = 1
    num_buffers: int = 1

    experimental_gemm_sms: int | None = None
    experimental_local_comm_sms: int | None = None

    gemm_tuned: bool = False

    def __post_init__(self):
        for value in (self.experimental_gemm_sms, self.experimental_local_comm_sms):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(
                    "Experimental SM budgets must be positive integers or None"
                )
        if self.experimental_gemm_sms is not None and self.gemm_tuned:
            raise ValueError("Headroom experiment requires fixed GEMM policies")
        if type(self.num_chunks) is not int or self.num_chunks < 1:
            raise ValueError("num_chunks must be a positive integer")
        if (
            type(self.num_buffers) is not int
            or not 1 <= self.num_buffers <= self.num_chunks
        ):
            raise ValueError("Require 1 <= num_buffers <= num_chunks")
        for name in (
            "ep_size",
            "num_experts",
            "top_k",
            "model_dim",
            "expert_hidden_dim",
            "token_padding",
            "num_comm_sms",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.tokens_per_rank is not None and (
            type(self.tokens_per_rank) is not int or self.tokens_per_rank <= 0
        ):
            raise ValueError(
                "Legacy tokens_per_rank must be a positive integer or None"
            )
        if self.num_experts % self.ep_size:
            raise ValueError("E must be divisible by G")
        if self.ep_size > 128 or self.top_k > min(32, self.num_experts):
            raise ValueError(
                "This MoE integration requires G <= 128 and K <= min(E,32)"
            )
        if self.latent_dim is not None and (
            type(self.latent_dim) is not int or self.latent_dim <= 0
        ):
            raise ValueError("latent_dim must be a positive integer")
        if type(self.num_shared_experts) is not int or self.num_shared_experts < 0:
            raise ValueError("num_shared_experts must be a nonnegative integer")
        if self.shared_expert_dim is not None and (
            type(self.shared_expert_dim) is not int or self.shared_expert_dim <= 0
        ):
            raise ValueError("shared_expert_dim must be a positive integer")
        if self.feature_dim % 128 or self.expert_hidden_dim % 128:
            raise ValueError("Expert feature/hidden widths must be multiples of 128")
        for name in (
            "overlap",
            "renormalize_topk",
            "profile_ranges",
            "retain_intermediates",
            "reuse_communication_buffers",
            "share_expert_banks",
            "fuse_probability_backward",
            "fuse_input_grad_accumulation",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        if (
            self.weight_grad_dtype is not None
            and self.parameter_dtype is not None
            and self.weight_grad_dtype != self.parameter_dtype
        ):
            raise ValueError(
                "weight_grad_dtype and its legacy parameter_dtype alias disagree"
            )
        grad_dtype = self.weight_grad_dtype or self.parameter_dtype or torch.float32
        if grad_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("weight_grad_dtype must be FP32 or BF16")
        object.__setattr__(self, "weight_grad_dtype", grad_dtype)
        object.__setattr__(self, "parameter_dtype", grad_dtype)
        if self.gradient_output_mode not in ("owned", "copy"):
            raise ValueError("gradient_output_mode must be owned or copy")
        if self.weight_transport is None:
            object.__setattr__(
                self,
                "weight_transport",
                "bf16" if self.compute_precision == "bf16" else "fp8",
            )
        if self.weight_transport not in ("bf16", "fp8"):
            raise ValueError("weight_transport must be bf16 or fp8")
        if self.compute_precision not in ("bf16", "fp8_current"):
            raise ValueError(
                "QuACK SM90 supports bf16 or fp8_current here; block scaling is not implemented"
            )
        if self.activation_transport not in ("bf16", "fp8"):
            raise ValueError("activation_transport must be bf16 or fp8")
        if self.activation_transport == "fp8" and self.compute_precision == "bf16":
            raise ValueError("FP8 activation transport requires FP8 expert computation")
        if (self.weight_transport == "fp8") != (self.compute_precision != "bf16"):
            raise ValueError(
                "FP8 computation and FP8 transport must be selected together"
            )
        if self.router_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("router_dtype must be FP32 or BF16")
        if self.router_weight_grad_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("router_weight_grad_dtype must be FP32 or BF16")
        if (
            self.router_dtype == torch.float32
            and self.router_weight_grad_dtype != torch.float32
        ):
            raise ValueError("An FP32 router requires FP32 gradients")
        if not math.isfinite(self.init_std) or self.init_std <= 0:
            raise ValueError("init_std must be finite and positive")

        if type(self.gemm_tuned) is not bool:
            raise ValueError("gemm_tuned must be bool")
        if self.latent_dim is not None:
            raise ValueError(
                "Latent projections are not yet implemented in this experimental version"
            )
        if not self.fuse_probability_backward or not self.fuse_input_grad_accumulation:
            raise ValueError(
                "Packed QuACK expert math requires both fusion flags enabled"
            )
        if self.retain_intermediates:
            raise ValueError(
                "Intermediate retention is not yet implemented in the QuACK prototype"
            )

    @property
    def local_experts(self):
        return self.num_experts // self.ep_size

    @property
    def feature_dim(self):
        return self.model_dim

    @property
    def shared_width(self):
        return self.num_shared_experts * (
            self.shared_expert_dim or self.expert_hidden_dim
        )

    @property
    def dispatched_rows(self):
        if self.tokens_per_rank is None:
            raise ValueError("Dispatched shape is known after binding a MoonEP buffer")
        return self.tokens_per_rank * self.top_k + 2 * self.local_experts * (
            self.token_padding - 1
        )


def config_signature(c):
    return json.dumps({"abi": 6, "config": asdict(c)}, sort_keys=True, default=str)


def _configuration(spec):
    record = json.loads(spec)
    if record["abi"] != 6:
        raise ValueError("QuACK MoE ABI mismatch")
    cfg = record["config"]
    types = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16}
    for name in (
        "parameter_dtype",
        "weight_grad_dtype",
        "router_dtype",
        "router_weight_grad_dtype",
    ):
        if name in cfg:
            cfg[name] = types[cfg[name]]
    return MoEConfig(**cfg)
