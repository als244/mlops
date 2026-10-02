"""Optional backends must not affect ordinary MLOps imports or CPU config use."""

import os
import subprocess
import sys

import pytest
import torch

from mlops.expert_parallel import LoRAConfig, QuackMoEConfig, TEMoEConfig


@pytest.mark.parametrize("config_type", [QuackMoEConfig, TEMoEConfig])
def test_config_validates_expert_ownership_and_precisions(config_type):
    options = {
        "ep_size": 2,
        "num_experts": 8,
        "top_k": 2,
        "model_dim": 512,
        "expert_hidden_dim": 1024,
    }
    config = config_type(**options, compute_precision="fp8_current")
    assert config.local_experts == 4
    assert config.weight_grad_dtype == torch.float32
    with pytest.raises(ValueError):
        config_type(**dict(options, num_experts=7))
    with pytest.raises(ValueError):
        config_type(**dict(options, top_k=9))
    with pytest.raises(ValueError):
        config_type(**options, compute_precision="unknown")


def test_lora_defaults_and_invalid_rank():
    config = LoRAConfig()
    assert config.rank == 32 and config.scale == 1
    assert config.compute_dtype == torch.bfloat16
    assert config.gradient_dtype == torch.float32
    with pytest.raises(ValueError):
        LoRAConfig(rank=0)


def test_cpu_import_does_not_load_optional_accelerator_packages():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'quack', 'sonicmoe', 'moonep', 'transformer_engine', 'quack_moe', 'te_moe', 'moe_lab', 'shadowspill'}:
            raise AssertionError('Unexpected dependency import: ' + fullname)
sys.meta_path.insert(0, Block())
import torch
import mlops
from mlops.expert_parallel import QuackMoEConfig, TEMoEConfig, LoRAConfig, create_buffer
from mlops.expert_parallel.parameters import BF16ComputeWeight
from mlops.expert_parallel.reference import expert_computation, route
from mlops.expert_parallel.reference.lora import expert_computation_lora
assert not torch.cuda.is_initialized()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )
