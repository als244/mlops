"""Chunk resources, buffer binding and saved-plan reconstruction.

Execution loops live in the scheduler, independently of these shared resources.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch

from .chunk_buffers import ChunkBufferPool
from .chunk_experts import ChunkExperts
from .communication import _PLAN_FIELDS
from .runtime_resources import RuntimeResources


@dataclass
class Chunk:
    index: int
    buffer: object
    plan: object
    cu: torch.Tensor
    slots: torch.Tensor
    received: torch.Tensor
    probabilities: torch.Tensor | None
    ready: object = None
    preact: torch.Tensor | None = None
    saved_x: torch.Tensor | None = None


class ChunkResources(RuntimeResources):
    def __init__(self, config, group, device, buffer):
        self.caller_buffer = buffer
        if not isinstance(buffer, ChunkBufferPool):
            if config.num_chunks != 1 or config.num_buffers != 1:
                raise ValueError(
                    "Multiple chunks require a caller-owned ChunkBufferPool"
                )
            buffer = ChunkBufferPool([buffer], num_chunks=1)
        if (
            buffer.num_chunks != config.num_chunks
            or len(buffer.buffers) != config.num_buffers
        ):
            raise ValueError("Chunk configuration and caller buffer pool disagree")
        if config.tokens_per_rank % config.num_chunks:
            raise ValueError("Token count must be divisible by num_chunks")
        super().__init__(config, group, device, buffer.buffers[0])
        self.buffer = buffer
        self.chunk_cfg = replace(
            config, tokens_per_rank=config.tokens_per_rank // config.num_chunks
        )
        from .chunk_fp8_experts import ChunkFP8Experts

        math_type = (
            ChunkExperts if config.compute_precision == "bf16" else ChunkFP8Experts
        )
        self.math = math_type(tuned=config.gemm_tuned, model_config=self.chunk_cfg)
        self.buffers = buffer.buffers
        from .chunk_headroom import configure

        configure(self)
        self.external_storage_extents += tuple(
            (
                b.hidden_nvsh_buffer_view.untyped_storage().data_ptr(),
                b.hidden_nvsh_buffer_view.untyped_storage().nbytes(),
            )
            for b in self.buffers[1:]
        )

    @contextmanager
    def execution(self):
        from .chunk_headroom import gemm_budget

        with super().execution(), gemm_budget(self.cfg.experimental_gemm_sms):
            yield

    def _chunk_plan(self, state):
        c = self.chunk_cfg
        return self.plan_type(
            **dict(zip(_PLAN_FIELDS, state[:7])),
            N=c.tokens_per_rank * c.top_k,
            R=c.ep_size,
            E=c.num_experts,
            B=c.local_experts,
            NvS=c.dispatched_rows,
            K=c.top_k,
        )

    def _prefetch(self, streams, chunk, phase):
        with streams.range(f"{phase}.{chunk.index}.prefetch", communication=True):
            for bank in self.banks:
                bank.weight_state.prefetch_slots(chunk.slots)
            chunk.ready = streams.comm.record_event()
