"""Select one device per torchrun worker before loading accelerator packages."""

import os
from datetime import timedelta


def select_rank_device():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank < 0:
        raise RuntimeError(f"LOCAL_RANK must be nonnegative, got {rank}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = str(rank)
    else:
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        if not 0 <= rank < len(devices):
            raise RuntimeError(
                f"LOCAL_RANK={rank} needs a device, but CUDA_VISIBLE_DEVICES={visible!r}"
            )
        selected = devices[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected


def initialize_group():
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("Expert-parallel GPU checks require an H100/SM90 device")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", 0), timeout=timedelta(minutes=5)
    )
