"""Low-rank expert parameters, independently configured from base precision."""

import json
import math
from dataclasses import asdict, dataclass

import torch


def packed_pitch(elements, *, element_size, experts, granularity, tile_elements):
    """Round a factor bank's pitch to both allocation and kernel alignment."""
    alignment = math.lcm(
        tile_elements, granularity // math.gcd(granularity, experts * element_size)
    )
    return math.ceil(elements / alignment) * alignment


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 32
    alpha: float = 32.0
    compute_dtype: torch.dtype = torch.bfloat16
    gradient_dtype: torch.dtype = torch.float32
    initialization_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if type(self.rank) is not int or self.rank < 16 or self.rank % 16:
            raise ValueError("LoRA rank must be a positive multiple of 16")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if self.compute_dtype != torch.bfloat16:
            raise ValueError(
                "LoRA currently supports BF16 compute, independent of base precision"
            )
        if self.gradient_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("LoRA gradient_dtype must be FP32 or BF16")
        if self.initialization_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("LoRA initialization_dtype must be FP32 or BF16")

    @property
    def scale(self):
        return self.alpha / self.rank


def signature(base_spec, lora):
    return json.dumps(
        {"abi": 1, "base": base_spec, "lora": asdict(lora)}, sort_keys=True, default=str
    )


def parse_signature(spec):
    record = json.loads(spec)
    if record["abi"] != 1:
        raise ValueError("LoRA operator ABI mismatch")
    types = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16}
    values = record["lora"]
    for key in ("compute_dtype", "gradient_dtype", "initialization_dtype"):
        values[key] = types[values[key]]
    return record["base"], LoRAConfig(**values)
