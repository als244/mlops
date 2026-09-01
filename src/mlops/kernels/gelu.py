"""Aten tanh-GELU forward/backward used by the production GPT-2 path."""

from __future__ import annotations

import torch


def gelu_forward(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def gelu_backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.gelu_backward(grad_output, x, approximate="tanh")
