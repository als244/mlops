"""Distributed forward/backward scheduling and runtime lifetime."""

from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import asdict

import torch
import torch.distributed as dist
import triton
from torch import Tensor

from .communication import _PLAN_FIELDS, _STATE_LEN, _CudaPhases, _walk_tensors
from .experts import _TEBackend
from .kernels.pointwise import _Pointwise
from .registry import _MATH_STREAMS, _REGISTRY_LOCK
from .weights import _ExpertBank


def moon_to_te_counts(ends: Tensor, rank: int, E: int, q: int) -> Tensor:
    e = ends.to(torch.int64)
    starts = torch.cat((torch.zeros_like(e[:1]), e[:-1]))
    sizes = e - starts
    return torch.cat((sizes[rank * q : (rank + 1) * q], sizes[E : E + q])).contiguous()


class _ScheduledMath:
    """Shared production orchestration, also exercised by CPU test doubles."""

    def _publish(self, params, zero=False):
        for bank, w in zip(self.banks, params):
            bank.publish(w)
            if zero:
                bank.prepare_grad_scratch()

    def _plan(self, state):
        c = self.cfg
        return self.plan_type(
            **dict(zip(_PLAN_FIELDS, state[:7])),
            N=c.tokens_per_rank * c.top_k,
            R=c.ep_size,
            E=c.num_experts,
            B=c.local_experts,
            NvS=c.dispatched_rows,
            K=c.top_k,
        )

    def _linear(self, index, x, counts, dgrad=False, *, accumulate_into=None):
        out = accumulate_into
        if out is None and self.reuse_communication_buffers and index == 0 and dgrad:
            out = self.buffer.hidden_nvsh_buffer_view
        return self.te.linear(
            x,
            self.banks[index].weight_views,
            counts,
            dgrad=dgrad,
            out=out,
            accumulate=accumulate_into is not None,
        )

    def _wg(self, index, x, dy, counts):
        self.te.wgrad(x, dy, self.banks[index].grad_views, counts)

    def _dispatch(self, x, p=None, ids=None, hist=None, plan=None):
        # async_finish=False means use the CURRENT stream: the phase driver has
        # already selected the communication stream. This is not CPU synchronization.
        return self.buffer.dispatch(
            x,
            p,
            ids,
            hist,
            plan=plan,
            async_finish=False,
            inter_rank_sync=False,
            zero_copy=self.reuse_communication_buffers and plan is not None,
            router_weights_zero_copy=False,
        )

    def _combine(self, plan, x, p=None):
        return self.buffer.combine(
            plan=plan,
            hidden_nvsh=x,
            route_weights_nvs=p,
            async_finish=False,
            inter_rank_sync=False,
            zero_copy=self.reuse_communication_buffers,
            router_weights_zero_copy=False,
        )

    def forward(self, x, p, ids, wg, wu, wd):
        with self.execution():
            params = (wg, wu, wd)
            self._check_inputs(x, p, ids, params)
            c = self.cfg
            hist = torch.zeros(c.num_experts, dtype=torch.int32, device=x.device)
            hist.scatter_add_(
                0, ids.flatten().to(torch.int64), torch.ones_like(ids.flatten())
            )
            s = self.make_phases((x, p, ids, hist, params))
            _, dispatched = s.phase(
                "fwd.dispatch_publish",
                compute=lambda: self._publish(params),
                communication=lambda: self._dispatch(x, p, ids, hist),
            )
            xp, pp, ends, plan = dispatched
            dispatched = hist = None

            def prepare():
                counts = moon_to_te_counts(
                    ends, self.rank, c.num_experts, c.local_experts
                )
                self.pw.mask_tail(xp, ends[-1:])
                self.pw.mask_tail(pp, ends[-1:])
                return counts

            counts, _ = s.phase(
                "fwd.gate_prefetch",
                compute=prepare,
                communication=lambda: self.banks[0].prefetch(plan),
            )
            prepare = ends = None
            # Gate compute cannot read up weights; independent work overlaps.
            gate, _ = s.phase(
                "fwd.gate_up_overlap",
                compute=lambda: self._linear(0, xp, counts),
                communication=lambda: self.banks[1].prefetch(plan),
            )

            def up_and_activation():
                up = self._linear(1, xp, counts)
                return up, self.pw.swiglu(gate, up)

            (up, h), _ = s.phase(
                "fwd.up_down_overlap",
                compute=up_and_activation,
                communication=lambda: self.banks[2].prefetch(plan),
            )
            up_and_activation = None

            def output():
                raw = self._linear(2, h, counts)
                destination = (
                    self.buffer.hidden_nvsh_buffer_view
                    if self.reuse_communication_buffers
                    else None
                )
                return raw, self.pw.scale(raw, pp, out=destination)

            (raw, weighted), _ = s.phase("fwd.expert_output", compute=output)
            output = h = None
            _, combined = s.phase(
                "fwd.combine", communication=lambda: self._combine(plan, weighted)
            )
            y, _, _ = combined
            combined = weighted = None
            state = [getattr(plan, f) for f in _PLAN_FIELDS] + [
                counts,
                xp,
                pp,
                gate,
                up,
                raw,
            ]
            s.phase("fwd.exit")
            return y, state

    def backward(self, dy, wg, wu, wd, state):
        with self.execution():
            if len(state) != _STATE_LEN:
                raise ValueError("Saved-state ABI mismatch")
            params = (wg, wu, wd)
            plan = self._plan(state)
            counts, xp, pp = state[7:10]
            dy = dy.contiguous()
            s = self.make_phases((dy, params, state))
            _, dispatched = s.phase(
                "bwd.dispatch_publish",
                compute=lambda: self._publish(params, zero=True),
                communication=lambda: self._dispatch(dy, plan=plan),
            )
            dyp = dispatched[0]
            dispatched = None
            gate, up, raw = state[10:13]
            h, _ = s.phase(
                "bwd.down_prefetch",
                compute=lambda: self.pw.swiglu(gate, up),
                communication=lambda: self.banks[2].prefetch(plan),
            )

            def down_backward():
                self.pw.mask_tail(dyp, counts.sum().reshape(1))
                if self.cfg.fuse_probability_backward:
                    dp_packed, de = self.pw.probability_grad_and_scale(
                        dyp, raw, pp, inplace=not self.cfg.retain_intermediates
                    )
                else:
                    dp_packed = self.pw.probability_grad(dyp, raw)
                    de = self.pw.scale(
                        dyp, pp, inplace=not self.cfg.retain_intermediates
                    )
                self._wg(2, h, de, counts)
                dh = self._linear(2, de, counts, dgrad=True)
                return dp_packed, dh

            (dp_packed, dh), _ = s.phase(
                "bwd.down_grad_gate_prefetch",
                compute=down_backward,
                communication=lambda: self.banks[0].prefetch(plan),
            )
            down_backward = h = dyp = raw = pp = None

            # The just-completed GROUP JOIN establishes down-gradient readiness on
            # EVERY rank, before any reducer is allowed to remote-read those slots.
            def gate_backward():
                dg, du = self.pw.swiglu_backward(gate, up, dh)
                self._wg(0, xp, dg, counts)
                return dg, du

            def reduce_down_and_prefetch_up():
                self.banks[2].reduce(plan, self.ctx)
                self.banks[1].prefetch(plan)

            (dg, du), _ = s.phase(
                "bwd.gate_grad_down_reduce_overlap",
                compute=gate_backward,
                communication=reduce_down_and_prefetch_up,
            )
            gate_backward = reduce_down_and_prefetch_up = gate = up = dh = None

            # Gate-gradient producers are now globally complete; up and gate have
            # disjoint gradient storage. The latter may be read/cleared concurrently.
            def up_wgrad_and_gate_dx():
                self._wg(1, xp, du, counts)
                return self._linear(0, dg, counts, dgrad=True)

            dxg, _ = s.phase(
                "bwd.up_grad_gate_reduce_overlap",
                compute=up_wgrad_and_gate_dx,
                communication=lambda: self.banks[0].reduce(plan, self.ctx),
            )
            up_wgrad_and_gate_dx = xp = dg = None

            def finish_dx():
                if self.fuse_input_grad_accumulation:
                    return self._linear(1, du, counts, dgrad=True, accumulate_into=dxg)
                dxu = self._linear(1, du, counts, dgrad=True)
                # Both contributions are BF16. Elementwise add performs the sum
                # with float opmath and rounds once, without full-size FP32 copies.
                return dxg + dxu if self.cfg.retain_intermediates else dxg.add_(dxu)

            dxp, _ = s.phase(
                "bwd.input_grad_up_reduce_overlap",
                compute=finish_dx,
                communication=lambda: self.banks[1].reduce(plan, self.ctx),
            )
            finish_dx = dxg = du = None
            # All projection gradients are reduced before ownership is transferred.
            grads, combined = s.phase(
                "bwd.combine",
                compute=lambda: tuple(
                    b.grad_result(w) for b, w in zip(self.banks, params)
                ),
                communication=lambda: self._combine(plan, dxp, dp_packed),
            )
            dx, dp, _ = combined
            combined = dxp = dp_packed = None
            s.phase("bwd.exit")
            return dx, dp, *grads


class _MoonRuntime(_ScheduledMath):
    def __init__(self, c, group, device, buffer):
        if not torch.cuda.is_available() or triton is None:
            raise RuntimeError("Production runtime requires CUDA and Triton")
        if not dist.is_initialized() or dist.get_world_size(group) != c.ep_size:
            raise ValueError(
                "Initialize distributed and pass an EP group of the configured size"
            )
        if dist.get_backend(group) != "nccl":
            raise ValueError("MoonEP runtime requires an NCCL EP process group")
        if device.index != torch.cuda.current_device():
            raise ValueError("Set the current CUDA device first")
        # Mismatched phase structures are a deadlock, not a permissible specialization.
        descriptor = json.dumps(asdict(c), sort_keys=True, default=str)
        descriptors = [None] * c.ep_size
        dist.all_gather_object(descriptors, descriptor, group=group)
        if len(set(descriptors)) != 1:
            raise ValueError("All EP ranks must have identical MoE configuration")
        total_sms = torch.cuda.get_device_properties(device).multi_processor_count
        if c.num_comm_sms >= total_sms or c.gemm_sm_margin >= total_sms:
            raise ValueError(
                "Communication SM count / GEMM margin must leave compute resources"
            )
        configured = os.environ.get("NVTE_EXT_MARGIN_SM")
        if configured is not None and configured != str(c.gemm_sm_margin):
            raise ValueError(
                "NVTE_EXT_MARGIN_SM conflicts with gemm_sm_margin; make them agree before initialization"
            )
        # TE reads this process-global hint in its grouped-tensor wrapper. It is
        # an algo-selection hint, NOT a hardware-enforced SM partition.
        os.environ["NVTE_EXT_MARGIN_SM"] = str(c.gemm_sm_margin)
        from moonep import Buffer
        from moonep.inter_rank_sync import launch_inter_rank_sync
        from moonep.planning import MoonEPCommPlan

        if set(MoonEPCommPlan.__dataclass_fields__) != set(_PLAN_FIELDS) | {
            "N",
            "R",
            "E",
            "B",
            "NvS",
            "K",
        }:
            raise RuntimeError("MoonEP plan ABI differs from this source integration")
        for n in (
            "async_finish",
            "inter_rank_sync",
            "zero_copy",
            "router_weights_zero_copy",
        ):
            if n not in inspect.signature(Buffer.dispatch).parameters:
                raise RuntimeError(f"MoonEP dispatch lacks {n}")
        self.cfg, self.group, self.device = c, group, device
        self.reuse_communication_buffers = (
            c.reuse_communication_buffers and not c.retain_intermediates
        )
        self.fuse_input_grad_accumulation = (
            c.fuse_input_grad_accumulation and not c.retain_intermediates
        )
        self.rank = dist.get_rank(group)
        self.overlap_enabled = c.overlap
        self.plan_type, self.rank_sync = MoonEPCommPlan, launch_inter_rank_sync
        self.closed = False
        self.retained_intermediates = None
        self._caller_stream = None
        self.te, self.pw = _TEBackend(c, device), _Pointwise()
        self.buffer = buffer
        self.ctx = self.buffer._require_ctx()
        if tuple(self.buffer.hidden_nvsh_buffer_view.shape) != (
            c.dispatched_rows,
            c.feature_dim,
        ):
            raise RuntimeError(
                "MoonEP static receive shape no longer matches this MoE integration"
            )
        self.banks = [
            self._make_bank(out_features, in_features)
            for out_features, in_features in self._bank_shapes(c)
        ]
        external = [self.buffer.hidden_nvsh_buffer_view]
        external.extend(t for bank in self.banks for t in bank.external_tensors())
        self.external_storage_extents = tuple(
            (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            for t in external
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def _make_bank(self, out_features, in_features):
        return _ExpertBank(
            self.cfg, self.rank, self.group, out_features, in_features, self.device
        )

    def _bank_shapes(self, config):
        return [(config.expert_hidden_dim, config.feature_dim)] * 2 + [
            (config.feature_dim, config.expert_hidden_dim)
        ]

    def make_phases(self, inputs):
        return _CudaPhases(self, inputs)

    @contextmanager
    def execution(self):
        if self.closed:
            raise RuntimeError("Runtime is closed")
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError("Wrong current CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CUDA graphs are not validated. Use options={'triton.cudagraphs': False}"
            )
        stream = torch.cuda.current_stream(self.device).cuda_stream
        with _REGISTRY_LOCK:
            previous = _MATH_STREAMS.setdefault(self.device.index, stream)
            if previous != stream:
                raise RuntimeError(
                    "All TE kernels for this MoE integration on a GPU must share one compute stream; TE caches shared workspace"
                )
        if self._caller_stream is None:
            self._caller_stream = stream
        elif self._caller_stream != stream:
            raise RuntimeError(
                "Use one caller/compute stream; the runtime owns its separate comm stream"
            )
        yield

    def _check_inputs(self, x, p, ids, params):
        c = self.cfg
        if x.shape != (c.tokens_per_rank, c.feature_dim) or x.dtype != torch.bfloat16:
            raise ValueError(
                "Expert input must have fixed [S, expert_width] BF16 shape"
            )
        if (
            p.shape != (c.tokens_per_rank, c.top_k)
            or p.dtype != torch.float32
            or ids.shape != p.shape
            or ids.dtype != torch.int32
        ):
            raise ValueError(
                "Routing requires FP32 probabilities and INT32 IDs of shape [S,K]"
            )
        flat_weights = tuple(_walk_tensors(params))
        if any(t.device != self.device or not t.is_contiguous() for t in (x, p, ids)):
            raise ValueError(
                "Activation and routing inputs must be contiguous on the runtime device"
            )
        if any(t.device != self.device for t in flat_weights):
            raise ValueError("Compute weight components must be on the runtime device")
        if c.compute_precision == "bf16":
            for w, b in zip(params, self.banks):
                if (
                    w.shape != (c.local_experts, b.out_features, b.in_features)
                    or w.dtype != torch.bfloat16
                ):
                    raise ValueError("BF16 compute component shape/dtype mismatch")
                if (
                    not w.is_contiguous()
                    and w.stride() != b.weight_state.parameter_data.stride()
                ):
                    raise ValueError(
                        "BF16 component must be contiguous or use the declared expert padding"
                    )
        else:
            self._check_fp8_components(params)

    def _check_fp8_components(self, params):
        from mlops.expert_parallel.transformer_engine.parameters.components import (
            component_names,
        )

        names = component_names(self.cfg.compute_precision)
        # A relocated input may be copied, but a permutation/view into provider
        # destinations could overwrite another still-live input during publish.
        extents = [
            (tensor.data_ptr(), tensor.numel() * tensor.element_size())
            for bank in self.banks
            for component in bank.components.values()
            for tensor in (component.remote, component.replicas)
        ]
        for bank, weights in zip(self.banks, params, strict=True):
            bank.validate_components(weights)
            for index, fields in enumerate(weights):
                for name, value in zip(names, fields, strict=True):
                    destination = bank.components[name].views[index]
                    pointer = value.data_ptr()
                    end = pointer + value.numel() * value.element_size()
                    if pointer != destination.data_ptr() and any(
                        pointer < start + size and start < end
                        for start, size in extents
                    ):
                        raise ValueError(
                            "FP8 input aliases a different provider destination; "
                            "use an independent component allocation"
                        )

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.retained_intermediates = None
        self.banks.clear()
        self.closed = True
