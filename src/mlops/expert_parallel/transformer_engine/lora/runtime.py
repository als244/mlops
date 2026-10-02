"""TE expert math with frozen base weights and grouped BF16 LoRA branches."""

from dataclasses import replace

import torch
import torch.distributed as dist
import transformer_engine_torch as tex

from ..communication import _PLAN_FIELDS
from ..config import config_signature
from ..experts import _TEBackend
from ..runtime import _MoonRuntime, moon_to_te_counts
from ..weights import _ExpertBank
from .config import signature
from .storage import FactorBank


class LoRARuntime(_MoonRuntime):
    def __init__(self, config, lora, group, device, buffer):
        self.lora = lora
        specs = [None] * config.ep_size
        dist.all_gather_object(
            specs, signature(config_signature(config), lora), group=group
        )
        if len(set(specs)) != 1:
            raise ValueError("Every EP rank must use the same LoRA configuration")
        super().__init__(config, group, device, buffer)
        self.low = _TEBackend(
            replace(config, compute_precision="bf16", weight_transport="bf16")
        )
        self.factors = [
            FactorBank(
                config, lora, self.rank, group, b.out_features, b.in_features, device
            )
            for b in self.banks
        ]
        self.alpha = torch.full(
            (1,), float(lora.scale), dtype=torch.float32, device=device
        )
        self.external_storage_extents += tuple(
            (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            for bank in self.factors
            for t in bank.external_tensors()
        )

    def _bank_shapes(self, config):
        return [
            (2 * config.expert_hidden_dim, config.feature_dim),
            (config.feature_dim, config.expert_hidden_dim),
        ]

    def _make_bank(self, out_features, in_features):
        return _ExpertBank(
            self.cfg,
            self.rank,
            self.group,
            out_features,
            in_features,
            self.device,
            trainable=False,
        )

    def _base_parameters(self, base):
        if self.cfg.compute_precision == "bf16":
            return base
        from ..parameters.components import component_names

        n = len(component_names(self.cfg.compute_precision))
        q = self.cfg.local_experts
        return [
            [base[(j * q + i) * n : (j * q + i + 1) * n] for i in range(q)]
            for j in range(2)
        ]

    def _publish_lora(self, base, factors):
        self._publish(self._base_parameters(base))
        for index, bank in enumerate(self.factors):
            bank.publish(factors[2 * index : 2 * index + 2])

    def _prefetch_lora(self, index, plan):
        self.banks[index].prefetch(plan)
        self.factors[index].prefetch(plan)

    def _projection(self, index, x, counts):
        base = self.te.linear(x, self.banks[index].weight_views, counts)
        a, b = self.factors[index].compute_weights
        z = self.low.linear(x, list(a.unbind()), counts)
        return self.low.linear(
            z, list(b.unbind()), counts, out=base, accumulate=True, alpha=self.alpha
        )

    def _projection_backward(self, index, x, dy, counts, *, out=None):
        bank = self.factors[index]
        a, b = bank.compute_weights
        av, bv = list(a.unbind()), list(b.unbind())
        dx = self.te.linear(
            dy, self.banks[index].weight_views, counts, dgrad=True, out=out
        )
        dz = self.low.linear(dy, bv, counts, dgrad=True, alpha=self.alpha)
        self.low.linear(dz, av, counts, dgrad=True, out=dx, accumulate=True)
        z = self.low.linear(x, av, counts)
        da = [*bank.grad_views[0][0].unbind(), *bank.grad_views[1][0].unbind()]
        db = [*bank.grad_views[0][1].unbind(), *bank.grad_views[1][1].unbind()]
        self.low.wgrad(x, dz, da, counts)
        self.low.wgrad(z, dy, db, counts, alpha=self.alpha)
        return dx

    def lora_forward(self, x, p, ids, base, factors, hist, shared):
        if shared:
            raise ValueError(
                "The TE shared expert uses its separate GEMM/SwiGLU operators"
            )
        with self.execution():
            self._check_inputs(x, p, ids, self._base_parameters(base))
            phases = self.make_phases((x, p, ids, base, factors, hist))
            _, dispatched = phases.phase(
                "fwd.dispatch_publish",
                compute=lambda: self._publish_lora(base, factors),
                communication=lambda: self._dispatch(x, p, ids, hist),
            )
            xp, pp, ends, plan = dispatched
            dispatched = None

            def prepare():
                counts = moon_to_te_counts(
                    ends, self.rank, self.cfg.num_experts, self.cfg.local_experts
                )
                self.pw.mask_tail(xp, counts.sum().reshape(1))
                self.pw.mask_tail(pp, counts.sum().reshape(1))
                return counts

            counts, _ = phases.phase(
                "fwd.gate_up_prefetch",
                compute=prepare,
                communication=lambda: self._prefetch_lora(0, plan),
            )
            prepare = ends = None

            def gate_up():
                pre = self._projection(0, xp, counts)
                return pre, tex.swiglu(pre, None)

            (pre, h), _ = phases.phase(
                "fwd.gate_up",
                compute=gate_up,
                communication=lambda: self._prefetch_lora(1, plan),
            )
            gate_up = None
            raw, _ = phases.phase(
                "fwd.down", compute=lambda: self._projection(1, h, counts)
            )
            h = None
            destination = (
                self.buffer.hidden_nvsh_buffer_view
                if self.reuse_communication_buffers
                else None
            )
            weighted, _ = phases.phase(
                "fwd.route_scale",
                compute=lambda: self.pw.scale(raw, pp, out=destination),
            )
            _, combined = phases.phase(
                "fwd.combine", communication=lambda: self._combine(plan, weighted)
            )
            y = combined[0]
            state = [getattr(plan, key) for key in _PLAN_FIELDS] + [
                counts,
                xp,
                pp,
                pre,
                raw,
            ]
            phases.phase("fwd.exit")
            return y, state

    def lora_backward(self, dy, base, factors, state, x, shared):
        with self.execution():
            if len(state) != 12:
                raise ValueError("TE LoRA saved-state ABI mismatch")
            plan = self._plan(state)
            counts, xp, pp, pre, raw = state[7:]
            phases = self.make_phases((dy, base, factors, state))

            def publish():
                self._publish_lora(base, factors)
                for bank in self.factors:
                    bank.prepare_grad_scratch()

            _, dispatched = phases.phase(
                "bwd.dispatch_publish",
                compute=publish,
                communication=lambda: self._dispatch(dy.contiguous(), plan=plan),
            )
            publish = None
            dyp = dispatched[0]
            dispatched = None
            h, _ = phases.phase(
                "bwd.down_prefetch",
                compute=lambda: tex.swiglu(pre, None),
                communication=lambda: self._prefetch_lora(1, plan),
            )

            def down_backward():
                self.pw.mask_tail(dyp, counts.sum().reshape(1))
                dp, de = self.pw.probability_grad_and_scale(dyp, raw, pp)
                return dp, self._projection_backward(1, h, de, counts)

            (dp, dh), _ = phases.phase(
                "bwd.down",
                compute=down_backward,
                communication=lambda: self._prefetch_lora(0, plan),
            )
            down_backward = h = dyp = raw = pp = None
            dpre, _ = phases.phase(
                "bwd.swiglu",
                compute=lambda: tex.dswiglu(dh, pre, None),
                communication=lambda: self.factors[1].reduce(plan, self.ctx),
            )
            dh = pre = None
            destination = (
                self.buffer.hidden_nvsh_buffer_view
                if self.reuse_communication_buffers
                else None
            )
            dxp, _ = phases.phase(
                "bwd.gate_up",
                compute=lambda: self._projection_backward(
                    0, xp, dpre, counts, out=destination
                ),
            )
            xp = dpre = None
            phases.phase(
                "bwd.gate_up_reduce",
                communication=lambda: self.factors[0].reduce(plan, self.ctx),
            )
            grads, combined = phases.phase(
                "bwd.combine",
                compute=lambda: [
                    g for bank in self.factors for g in bank.grad_result()
                ],
                communication=lambda: self._combine(plan, dxp, dp),
            )
            phases.phase("bwd.exit")
            return combined[0], combined[1], grads

    def close(self):
        super().close()
        self.factors.clear()
