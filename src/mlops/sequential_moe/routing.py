"""GPU route ordering and one kernel-published host table per forward."""

import torch
import triton

from .kernels import publish
from .profiling import annotate


@torch.library.custom_op("mlops_seqmoe::layout", mutates_args=())
def layout(ids: torch.Tensor, experts: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with annotate("fwd/routing_layout"):
        flat = ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.zeros(experts, dtype=torch.int64, device=ids.device)
        counts.scatter_add_(0, flat, torch.ones_like(flat))
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        host = torch.empty(experts + 1, dtype=torch.int64, pin_memory=True)
        publish[(1,)](offsets, host, experts + 1, triton.next_power_of_2(experts + 1))
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(ids.device))
        ready.synchronize()
        # Kernel completion orders GPU writes before CPU reads. No cudaMemcpy,
        # no global device synchronization, and no polling on host memory.
        # Each expert gets ordinary CPU data with the same shape and zero storage
        # offset. Views into the shared table would have different storage offsets,
        # which are part of a compiled task's storage contract. These 16-byte CPU
        # copies need no CUDA transfer or additional synchronization.
        bounds = [host[index : index + 2].clone() for index in range(experts)]
        return order, bounds


@layout.register_fake
def _layout_fake(ids, experts):
    return (
        torch.empty(ids.numel(), dtype=torch.int64, device=ids.device),
        [torch.empty(2, dtype=torch.int64, device="cpu") for _ in range(experts)],
    )
