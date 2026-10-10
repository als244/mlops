"""Load selected unmodified reference definitions from the pinned HF source."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def official(*names):
    path = Path(__file__).parent / "references/modeling_glm5_next.py"
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if len(selected) != len(names):
        raise RuntimeError(
            "Pinned reference source does not contain requested definitions"
        )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ns = {
        "torch": torch,
        "GradientCheckpointingLayer": nn.Module,
        "use_experts_implementation": lambda cls: cls,
        "ALL_ATTENTION_FUNCTIONS": SimpleNamespace(
            get_interface=lambda name, fallback: fallback
        ),
        "nn": nn,
        "F": F,
        "ACT2FN": {"silu": F.silu, "sigmoid": torch.sigmoid},
        "use_kernel_forward_from_hub": lambda *a, **k: lambda value: value,
        "use_kernel_func_from_hub_with_fallback": lambda *a, **k: lambda value: value,
        "use_kernelized_func": lambda *a, **k: lambda value: value,
        "force_accelerate_hooks": lambda *a, **k: lambda value: value,
    }
    # Execute only allowlisted definitions from the pinned, licensed test reference.
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)  # noqa: S102
    return tuple(ns[name] for name in names)
