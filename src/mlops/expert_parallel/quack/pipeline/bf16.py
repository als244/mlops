"""Optional chunk accumulation using Quack's existing GEMM output epilogue."""

from quack.gemm_interface import gemm_add_inplace, gemm_tuned

from ..experts.bf16 import QuackExperts, _gemm


class ChunkExperts(QuackExperts):
    def weight_gradient(self, x, dy, cu, bank, *, accumulate_home=False, offsets=None):
        if not accumulate_home:
            return super().weight_gradient(x, dy, cu, bank)
        config = self.policy.weight_gradient(x.shape[-1], dy.shape[-1])
        q = bank.cfg.local_experts
        for group_offsets, destination, add in (
            (cu[: q + 1], bank.local_home_grad, True),
            (cu[q:], bank.local_replica_grad, False),
        ):
            if group_offsets.data_ptr() % 16:
                group_offsets = group_offsets.clone()
            out = destination[:, : bank.out_features, :]
            if add:
                if config is None:
                    gemm_add_inplace(
                        dy.T, x, out, cu_seqlens_k=group_offsets, tuned=self.tuned
                    )
                else:
                    gemm_tuned.fn(
                        dy.T,
                        x,
                        out,
                        cu_seqlens_k=group_offsets,
                        config=config,
                        add_to_output=True,
                    )
            else:
                _gemm(
                    dy.T,
                    x,
                    out=out,
                    cu_seqlens_k=group_offsets,
                    tuned=self.tuned,
                    config=config,
                )
