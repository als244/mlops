"""Isolated SM90 launch experiment; no installed dependency files are edited.

The cap bounds persistent CTA counts, not physical SM affinity. This experiment
admits fixed SM90 policies whose cluster K dimension is one. Applying
it inside the runtime restores the policy in compiled graph-pair entrypoints.
"""

import hashlib
import importlib
import inspect
from contextlib import contextmanager
from contextvars import ContextVar

_budget = ContextVar("chunk_gemm_sms", default=None)
_original = None
_observed = set()


def bounded_clusters(hardware_clusters, cluster_size, num_sms):
    if hardware_clusters <= 0:
        raise ValueError("Headroom requires persistent GEMMs")
    if cluster_size < 1 or num_sms < cluster_size:
        raise ValueError("GEMM budget must fit a whole cluster")
    return min(hardware_clusters, num_sms // cluster_size)


def _scheduler(plan, *args, **kwargs):
    scheduler = _original(plan, *args, **kwargs)
    budget = _budget.get()
    if budget is None:
        return scheduler
    if type(plan).__module__ not in ("quack.gemm", "quack.gemm_runtime.host"):
        raise ValueError("Unvalidated Quack plan type for the headroom experiment")
    if plan.is_sm100_family:
        raise ValueError("Headroom experiment is scoped to SM90")
    # Cluster K=1 is checked against every selected fixed policy at init.
    cluster_size = plan.cluster_M * plan.cluster_N
    clusters = bounded_clusters(plan.max_active_clusters, cluster_size, budget)
    _observed.add(
        (type(plan).__name__, cluster_size, plan.max_active_clusters, clusters)
    )
    from cutlass import Int32

    # Never change the cached plan or its static scheduler. Preserve semaphores,
    # swizzle, and all other per-call scheduler values returned by stock Quack.
    return scheduler._replace(max_active_clusters=Int32(clusters))


def install():
    global _original
    if _original is not None:
        return
    utils = importlib.import_module("quack.gemm_tvm_ffi_utils")
    original = utils.plan_scheduler_args
    digest = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if digest != "6cc2e8a106171751d278cfaabc437ee21e1cec81bd6c6d8c20b3634011825120":
        raise RuntimeError(
            "Headroom experiment requires the pinned stock Quack scheduler"
        )
    modules = [utils] + [
        importlib.import_module(name)
        for name in ("quack.gemm", "quack.gemm_runtime.host")
    ]
    if any(module.plan_scheduler_args is not original for module in modules):
        raise RuntimeError("Quack scheduler was already changed")
    _original = original
    for module in modules:
        module.plan_scheduler_args = _scheduler


@contextmanager
def gemm_budget(num_sms):
    if num_sms is not None and (type(num_sms) is not int or num_sms < 1):
        raise ValueError("GEMM SM budget must be a positive integer or None")
    token = _budget.set(num_sms)
    try:
        yield
    finally:
        _budget.reset(token)


def configure(runtime):
    c = runtime.cfg
    gemm_sms, local_sms = c.experimental_gemm_sms, c.experimental_local_comm_sms
    if gemm_sms is None and local_sms is None:
        return
    import torch

    properties = torch.cuda.get_device_properties(runtime.device)
    if properties.major != 9:
        raise ValueError("Headroom experiment requires SM90")
    for value in (gemm_sms, local_sms):
        if value is not None and value > properties.multi_processor_count:
            raise ValueError("Requested SM budget exceeds device capacity")
    if gemm_sms is not None:
        policy = runtime.math.policy
        for name in (
            "up",
            "down",
            "down_backward",
            "input_gradient",
            "up_weight_gradient",
            "down_weight_gradient",
        ):
            config = getattr(policy, name)
            if config is None or config.cluster_k != 1:
                raise ValueError("Headroom requires a fixed policy with cluster_k=1")
            bounded_clusters(
                properties.multi_processor_count,
                config.cluster_m * config.cluster_n,
                gemm_sms,
            )
        install()
    if local_sms is not None:
        # MoonEP exposes this separately from its transfer-kernel num_sms.
        # These contexts belong to the caller-created experimental buffer pool.
        for buffer in runtime.buffers:
            buffer._require_ctx()["num_sms_dedup"] = local_sms


def observations():
    return [
        {
            "plan": kind,
            "cluster_size": size,
            "hardware_clusters": hardware,
            "launch_clusters": launch,
            "launch_ctas": size * launch,
        }
        for kind, size, hardware, launch in sorted(_observed)
    ]
