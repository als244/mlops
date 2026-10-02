"""Extend stock SM90 candidates without changing Quack's installed source.

Apply once before GEMM use. The stock tuner, pruning rules, timing procedure,
and exact-shape cache keys stay in control. Explicit GEMM configs still bypass
tuning. Other GPU architectures keep their original candidates.
"""

import hashlib
import inspect
import sys
import textwrap
from functools import lru_cache, wraps

_applied = False


def apply():
    global _applied
    if _applied:
        return False

    import quack.gemm_config as gc
    from quack.autotuner import AutotuneConfig

    original = gc._get_sm90_configs
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != (
        "2a25ec90d70f3f807f535fc1b13b61496f65ac71ebf2b4459ff8bd4ea4a6ac0a"
    ):
        raise RuntimeError(
            "Unsupported Quack candidate generator. Use the pinned stock release "
            "and import mlops.expert_parallel.quack before sonicmoe.functional."
        )
    interface = sys.modules.get("quack.gemm_interface")
    mod_tuning = sys.modules.get("quack.gemm_runtime.autotune")
    if (interface is not None and interface.gemm_tuned.cache) or (
        mod_tuning is not None and mod_tuning._MOD_TUNERS
    ):
        raise RuntimeError(
            "Import mlops.expert_parallel.quack before creating or using Quack GEMM tuners"
        )

    @wraps(original)
    def extended_sm90_configs(epilogue=None, tune_coop=True):
        candidates = list(original(epilogue, tune_coop))
        swaps = (False,) if epilogue in ("gated", "lse") else (False, True)
        for cm, cn in ((1, 2), (2, 1)):
            for swap in swaps:
                candidate = gc.GemmConfig(
                    tile_m=128,
                    tile_n=96,
                    pingpong=True,
                    cluster_m=cm,
                    cluster_n=cn,
                    swap_ab=swap,
                    device_capacity=9,
                    is_dynamic_persistent=False,
                    use_tma_gather=False,
                )
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    gc._get_sm90_configs = extended_sm90_configs
    # This tuner captures candidates at module import; mod/epilogue tuners are
    # created lazily and will call the extended generator themselves.
    if interface is not None:
        interface.gemm_tuned.configs = [
            AutotuneConfig(config=c) for c in gc.get_all_configs()
        ]
    _applied = True
    return True


@lru_cache(maxsize=1)
def router_kernels():
    """Load SonicMoE's router without adopting its global Quack tuning policy."""
    import quack.autotuner as at
    import quack.gemm_config as gc
    from quack.gemm_interface import gemm_tuned

    saved = (
        at.Autotuner.__call__,
        gc._get_sm90_configs,
        gc._get_sm100_configs,
        gemm_tuned.configs,
    )
    try:
        from sonicmoe.functional.backward import _topk_softmax_bwd
        from sonicmoe.functional.forward import _topk_softmax_fwd
        from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton
    finally:
        (
            at.Autotuner.__call__,
            gc._get_sm90_configs,
            gc._get_sm100_configs,
            gemm_tuned.configs,
        ) = saved
    return _topk_softmax_fwd, _topk_softmax_bwd, TC_topk_router_metadata_triton
