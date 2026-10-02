"""One chunk loop with optional pipelining of preparation and FIFO transfers.

A slot lives from dispatch preparation through combine's peer-release barrier.
Plans and saved activations are owned tensors; each slot is reused only after
its previous combine has finished on every rank. Expert arithmetic is unchanged.
"""

from dataclasses import dataclass

import torch
from moonep.combine import launch_combine
from moonep.combine_prologue import launch_combine_prologue
from moonep.dispatch import launch_dispatch
from moonep.planning import allocate_planning_outputs, launch_planning

from .activation_transport import mask_rows_tail, restore_rows, save_rows
from .communication import _PLAN_FIELDS
from .pipeline.resources import Chunk, ChunkResources
from .pipeline.transport import ChunkTransport
from .runtime_resources import local_group_counts
from .streams import Streams


@dataclass
class Pending:
    index: int
    buffer: object
    plan: object
    ends: object
    saved: object
    transferred: object
    scales: object = None


class _Runtime(ChunkResources):
    def _shared_math(self):
        from .shared_expert import compiled_shared_math

        return compiled_shared_math()

    def __init__(self, config, group, device, buffer):
        super().__init__(config, group, device, buffer)
        # One chunk uses the caller's existing MoonEP stream for preparation
        # and transfers. Only a pipeline needs an additional lane to overlap
        # the next preparation with a transfer. Compute stays on the caller
        # stream in both cases, preserving shared-expert/reduction overlap.
        self.transfer_stream = (
            torch.cuda.Stream(device=device, priority=-1)
            if config.overlap and config.num_chunks > 1
            else buffer._comm_stream
        )
        self.transport = ChunkTransport(
            self.buffers,
            config.activation_transport,
            borrow_backward=config.reuse_communication_buffers,
        )
        self.state_stride = 11 + int(config.activation_transport == "fp8")

    def _slots(self, plan):
        from .pipeline.quantize import group_slots

        return group_slots(plan.experts_to_copy[self.rank], self.cfg.local_experts)

    def _submit_dispatch(
        self, streams, index, x, p=None, ids=None, saved=None, hist=None
    ):
        c = self.chunk_cfg
        buffer = self.buffers[index % len(self.buffers)]
        ctx = self.transport.context(buffer)
        start, end = index * c.tokens_per_rank, (index + 1) * c.tokens_per_rank
        phase = "bwd" if saved is not None else "fwd"
        with streams.range(f"{phase}.{index}.dispatch_prepare", communication=True):
            self.rank_sync(ctx)
            payload, metadata, scales = self.transport.prepare(
                buffer,
                x[start:end],
                p[start:end] if p is not None else None,
                reuse_plan=saved is not None,
            )
            streams.record((payload, metadata, scales))
            if saved is None:
                ids_chunk = ids[start:end]
                histogram = hist if self.cfg.num_chunks == 1 else None
                if histogram is None:
                    histogram = torch.zeros(
                        c.num_experts, dtype=torch.int32, device=x.device
                    )
                    histogram.scatter_add_(
                        0,
                        ids_chunk.flatten().long(),
                        torch.ones_like(ids_chunk.flatten()),
                    )
                plan, ends = allocate_planning_outputs(ctx)
                launch_planning(ctx, ids_chunk.reshape(-1), histogram, ends, plan)
            else:
                plan, ends = self._chunk_plan(saved), None
            streams.record(plan)
            prepared = streams.comm.record_event()
        streams.transfer.wait_event(prepared)
        with streams.bulk(f"{phase}.{index}.dispatch_transfer"):
            launch_dispatch(
                ctx,
                payload,
                metadata,
                plan,
                build_dedup_map=saved is None,
                pdl_trigger=False,
            )
            transferred = streams.transfer.record_event()
        return Pending(index, buffer, plan, ends, saved, transferred, scales)

    def _finish_dispatch(self, streams, pending, *, retain_forward=True):
        i, buffer, plan = pending.index, pending.buffer, pending.plan
        ctx, c = self.transport.context(buffer), self.chunk_cfg
        phase = "bwd" if pending.saved is not None else "fwd"
        streams.comm.wait_event(pending.transferred)
        with streams.range(f"{phase}.{i}.dispatch_finish", communication=True):
            received, pp = self.transport.collect(
                buffer,
                plan,
                pending.scales,
                reuse_plan=pending.saved is not None,
                retain_forward=retain_forward,
            )
            if pending.saved is None:
                counts = local_group_counts(
                    pending.ends, self.rank, c.num_experts, c.local_experts
                )
                cu = torch.cat((torch.zeros_like(counts[:1]), counts.cumsum(0))).to(
                    torch.int32
                )
                self.pw.mask_tail(pp, cu[-1:])
                saved_x = preact = None
            else:
                cu, saved_data, pp, preact = pending.saved[7:11]
                saved_x = restore_rows(saved_data, pending.saved[11:])
            mask_rows_tail(self.pw, received, cu[-1:])
            chunk = Chunk(
                i,
                buffer,
                plan,
                cu,
                self._slots(plan),
                received,
                pp,
                preact=preact,
                saved_x=saved_x,
            )
            streams.record(chunk)
            self.rank_sync(ctx)
            # This protects subsequent expert writes to the same slot, including
            # against peers still reading the previous dispatch contents.
            chunk.ready = streams.comm.record_event()
        return chunk

    def _combine(self, streams, chunk, value, dp=None):
        c, ctx, i = self.chunk_cfg, chunk.buffer._require_ctx(), chunk.index
        phase = "bwd" if dp is not None else "fwd"
        copied = value.data_ptr() != ctx["hidden_buf_local"].data_ptr()
        if copied and self.reuse_communication_buffers:
            raise ValueError(
                "Combine input must be the corresponding communication slot"
            )
        streams.record(value)
        with streams.range(f"{phase}.{i}.combine_prepare", communication=True):
            if copied:
                ctx["hidden_buf_local"].copy_(value)
            if dp is not None:
                ctx["weights_buf_local"].copy_(dp.view(torch.int32))
            output = torch.empty(
                (c.tokens_per_rank, c.model_dim), device=value.device, dtype=value.dtype
            )
            output_p = (
                torch.empty(
                    (c.tokens_per_rank, c.top_k),
                    device=value.device,
                    dtype=torch.float32,
                )
                if dp is not None
                else None
            )
            launch_combine_prologue(ctx, chunk.plan, pdl_trigger=False)
            prepared = streams.comm.record_event()
            streams.record((output, output_p))
        streams.transfer.wait_event(prepared)
        with streams.bulk(f"{phase}.{i}.combine_transfer"):
            launch_combine(
                ctx, output, chunk.plan.dst, output_sk=output_p, pdl_launch=False
            )
            self.rank_sync(ctx)
            released = streams.transfer.record_event()
        return (output, output_p), released

    def _destination(self, chunk):
        slot = chunk.buffer.hidden_nvsh_buffer_view
        return slot if self.reuse_communication_buffers else torch.empty_like(slot)

    def forward(
        self, x, p, ids, hist, w1, w2, *, shared_weights=None, _return_state=True
    ):
        with self.execution():
            self._check_inputs(x, p, ids, (w1, w2))
            if (
                hist.shape != (self.cfg.num_experts,)
                or hist.dtype != torch.int32
                or hist.device != x.device
            ):
                raise ValueError(
                    "Routing histogram must be INT32 [E] on the input device"
                )
            streams = Streams(self, (x, p, ids, hist, w1, w2, shared_weights))
            n, m = self.cfg.num_chunks, len(self.buffers)
            pending, outputs, state = {}, [], []
            shared_output = None
            shared_state = []
            with streams.range("fwd.publish"):
                self._publish((w1, w2))
                published = streams.main.record_event()
            streams.comm.wait_event(published)
            with streams.range("fwd.publish_barrier", communication=True):
                self.rank_sync(self.buffers[0]._require_ctx())
            pending[0] = self._submit_dispatch(streams, 0, x, p, ids, hist=hist)
            if shared_weights is not None:
                with streams.range("fwd.shared_expert"):
                    shared_output, *shared_state = self._shared_math()[0](
                        x, *shared_weights
                    )
                    streams.record((shared_output, shared_state))
                    if not _return_state:
                        shared_state = []
            for i in range(1, m):
                pending[i] = self._submit_dispatch(streams, i, x, p, ids, hist=hist)
            chunk = self._finish_dispatch(
                streams, pending.pop(0), retain_forward=_return_state
            )
            self._prefetch(streams, chunk, "fwd")
            for i in range(n):
                # Submit independent communication before expert launches. In
                # FP8 backward, reading group offsets synchronizes the host;
                # waiting until after GEMM submission would delay this work.
                following = (
                    self._finish_dispatch(
                        streams, pending.pop(i + 1), retain_forward=_return_state
                    )
                    if i + 1 in pending
                    else None
                )
                streams.main.wait_event(chunk.ready)
                with streams.range(f"fwd.{i}.up"):
                    preact, activation = self.math.up(
                        chunk.received,
                        self.banks[0].weight_state.compute_weights,
                        chunk.cu,
                    )
                    streams.record((preact, activation))
                    if not _return_state:
                        # Gate/up is the last reader of borrowed X. Down writes
                        # the same slot next on this compute stream. No saved
                        # input or preactivation escapes this output-only call.
                        chunk.received = preact = None
                with streams.range(f"fwd.{i}.down"):
                    weighted = self.math.down(
                        activation,
                        self.banks[1].weight_state.compute_weights,
                        chunk.probabilities,
                        chunk.cu,
                        self._destination(chunk),
                    )
                    expert_done = streams.main.record_event()
                activation = None
                if _return_state:
                    saved_data, extra = save_rows(chunk.received)
                    state.extend(
                        [getattr(chunk.plan, key) for key in _PLAN_FIELDS]
                        + [chunk.cu, saved_data, chunk.probabilities, preact]
                        + extra
                    )
                streams.comm.wait_event(expert_done)
                if following is not None:
                    self._prefetch(streams, following, "fwd")
                combined, released = self._combine(streams, chunk, weighted)
                outputs.append(combined[0])
                if i + m < n:
                    streams.comm.wait_event(released)
                    pending[i + m] = self._submit_dispatch(
                        streams, i + m, x, p, ids, hist=hist
                    )
                if following is None and i + 1 in pending:
                    following = self._finish_dispatch(
                        streams, pending.pop(i + 1), retain_forward=_return_state
                    )
                    self._prefetch(streams, following, "fwd")
                chunk = following
                weighted = combined = preact = None
            streams.finish((outputs, state))
            with streams.range("fwd.output_join"):
                output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
                if shared_output is not None:
                    output = output + shared_output
                return output, state + shared_state

    def backward(self, dy, w1, w2, state, *, x=None, shared_weights=None):
        with self.execution():
            n, m = self.cfg.num_chunks, len(self.buffers)
            stride = self.state_stride
            shared_state = []
            if shared_weights is not None:
                if x is None or len(state) != n * stride + 3:
                    raise ValueError("Shared-expert backward input/state ABI mismatch")
                state, shared_state = state[: n * stride], state[n * stride :]
            if len(state) != n * stride:
                raise ValueError("Chunk saved-state ABI mismatch")
            streams = Streams(
                self, (dy, w1, w2, state, x, shared_weights, shared_state)
            )
            pending, outputs = {}, []
            shared_gradients = None
            offsets = [None] * n
            if self.cfg.compute_precision != "bf16" and getattr(
                self.math, "needs_host_offsets", True
            ):
                # These saved offsets already exist at entry. Read them once,
                # before enqueueing expert math, rather than synchronizing after
                # each down-dgrad and again between both weight gradients.
                with streams.range("bwd.offsets_to_host"):
                    offsets = (
                        torch.stack([state[i * stride + 7] for i in range(n)])
                        .cpu()
                        .tolist()
                    )
            with streams.range("bwd.publish"):
                self._publish((w1, w2))
                published = streams.main.record_event()
            streams.comm.wait_event(published)
            with streams.range("bwd.publish_barrier", communication=True):
                self.rank_sync(self.buffers[0]._require_ctx())
            pending[0] = self._submit_dispatch(streams, 0, dy, saved=state[:stride])
            if shared_weights is not None:
                with streams.range("bwd.shared_expert"):
                    shared_gradients = self._shared_math()[1](
                        dy, x, *shared_state, *shared_weights
                    )
                    streams.record(shared_gradients)
            for i in range(1, m):
                pending[i] = self._submit_dispatch(
                    streams, i, dy, saved=state[i * stride : (i + 1) * stride]
                )
            chunk = self._finish_dispatch(streams, pending.pop(0))
            self._prefetch(streams, chunk, "bwd")
            for i in range(n):
                following = (
                    self._finish_dispatch(streams, pending.pop(i + 1))
                    if i + 1 in pending
                    else None
                )
                streams.main.wait_event(chunk.ready)
                with streams.range(f"bwd.{i}.grad_outputs"):
                    if i == 0:
                        for bank in self.banks:
                            bank.prepare_grad_scratch()
                with streams.range(f"bwd.{i}.down_input_gradient"):
                    dpreact, aprime, dp = self.math.down_backward(
                        chunk.received,
                        self.banks[1].weight_state.compute_weights,
                        chunk.preact,
                        chunk.probabilities,
                        chunk.cu,
                    )
                    streams.record((dpreact, aprime, dp))
                with streams.range(f"bwd.{i}.down_weight_gradient"):
                    self.math.weight_gradient(
                        aprime,
                        chunk.received,
                        chunk.cu,
                        self.banks[1],
                        accumulate_home=i > 0,
                        offsets=offsets[i],
                    )
                    down_gradient_ready = streams.main.record_event()
                aprime = None
                streams.comm.wait_event(down_gradient_ready)
                with streams.range(f"bwd.{i}.down_reduce", communication=True):
                    self.rank_sync(chunk.buffer._require_ctx())
                    self.banks[1].reduce(chunk.plan, chunk.buffer._require_ctx())
                with streams.range(f"bwd.{i}.up_weight_gradient"):
                    self.math.weight_gradient(
                        chunk.saved_x,
                        dpreact,
                        chunk.cu,
                        self.banks[0],
                        accumulate_home=i > 0,
                        offsets=offsets[i],
                    )
                with streams.range(f"bwd.{i}.input_gradient"):
                    dxp = self.math.input_gradient(
                        dpreact,
                        self.banks[0].weight_state.compute_weights,
                        chunk.cu,
                        self._destination(chunk),
                    )
                    expert_done = streams.main.record_event()
                dpreact = None
                streams.comm.wait_event(expert_done)
                with streams.range(f"bwd.{i}.reduce_accumulate", communication=True):
                    self.rank_sync(chunk.buffer._require_ctx())
                    self.banks[0].reduce(chunk.plan, chunk.buffer._require_ctx())
                if following is not None:
                    self._prefetch(streams, following, "bwd")
                combined, released = self._combine(streams, chunk, dxp, dp)
                outputs.append(combined)
                if i + m < n:
                    streams.comm.wait_event(released)
                    pending[i + m] = self._submit_dispatch(
                        streams,
                        i + m,
                        dy,
                        saved=state[(i + m) * stride : (i + m + 1) * stride],
                    )
                if following is None and i + 1 in pending:
                    following = self._finish_dispatch(streams, pending.pop(i + 1))
                    self._prefetch(streams, following, "bwd")
                chunk = following
                dxp = dp = combined = None
            streams.finish(outputs)
            with streams.range("bwd.output_join"):
                gradients = [b.grad_result(w) for b, w in zip(self.banks, (w1, w2))]
                streams.record(gradients)
                dx = (
                    outputs[0][0]
                    if len(outputs) == 1
                    else torch.cat([v[0] for v in outputs])
                )
                dp = (
                    outputs[0][1]
                    if len(outputs) == 1
                    else torch.cat([v[1] for v in outputs])
                )
                if shared_gradients is not None:
                    dx = dx + shared_gradients[0]
                    return dx, dp, gradients, list(shared_gradients[1:])
                return dx, dp, gradients
