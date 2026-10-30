"""Cooperative, opt-in Quack candidate searches on a lockstep DP group.

Set MLOPS_QUACK_TUNE_DISTRIBUTED=1 before importing MLOps. Every rank must
encounter the same tuner misses in the same order; independent workers must
leave it disabled. Callers must synchronize the tuning activations before
entering the model, so all ranks produce the same routing and intermediates.
"""
from __future__ import annotations

import builtins
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import torch.distributed as dist


def check_disk_cache(tuner, tuning_key, configs, bench_fn):
    from quack import autotuner as quack

    from .sweeps import _failed, _positive_setting, native_allocator

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed Quack tuning requires an initialized DP group")
    rank, world = dist.get_rank(), dist.get_world_size()
    full_configs = list(configs)
    labels = [str(config) for config in full_configs]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate tuning configurations")
    name = tuner.fn.__name__
    identity = (name, str(tuning_key), labels)
    peers = [None] * world
    dist.all_gather_object(peers, identity)
    if any(peer != identity for peer in peers):
        raise RuntimeError(
            "distributed Quack tuning requires identical search order, shapes and "
            "candidate lists on every rank; use shared-cache tuning for independent workers"
        )
    cache_key = hashlib.sha256(
        "-".join([quack.VERSION, str(tuning_key), *labels]).encode("utf-8")
    ).hexdigest()
    cache = quack.FileCacheManager(quack._base32(cache_key))
    filename = f"{name[:150]}.autotune.json"
    lock = None
    try:
        decision = [None]
        if rank == 0:
            lock = (Path(cache.cache_dir) / "mlops-tuning.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = cache.get_file(filename)
            if path and not os.environ.get("QUACK_FORCE_CACHE_UPDATE"):
                rows = json.loads(Path(path).read_text())["configs_timings"]
                if not _failed(value for _, value in rows):
                    decision[0] = rows
        dist.broadcast_object_list(decision, src=0)
        started = time.monotonic()
        if decision[0] is not None:
            rows = decision[0]
            tuner.bench_time = 0
        else:
            warmup = _positive_setting(None, "MLOPS_QUACK_TUNE_WARMUP_MS", 50.0)
            timed = _positive_setting(None, "MLOPS_QUACK_TUNE_TIME_MS", 180.0)
            assigned = full_configs[rank::world]
            print(
                f"mlops quack: distributed tuning {name}: rank {rank}/{world} "
                f"has {len(assigned)}/{len(full_configs)} candidates; "
                f"{warmup:g} ms warmup + {timed:g} ms timed each",
                flush=True,
            )
            error = None
            local_rows = []
            try:
                if assigned:
                    # Quack's benchmark closure refers to this same pruned list.
                    configs[:] = assigned
                    try:
                        bench_fn()
                        local_rows = [
                            (str(config), timing)
                            for config, timing in tuner.configs_timings.items()
                        ]
                    finally:
                        configs[:] = full_configs
            except Exception as exc:  # noqa: BLE001 - report failure to all peers
                error = f"{type(exc).__name__}: {exc}"
            local_best = min(local_rows, key=lambda row: row[1]) if local_rows else None
            shards = [None] * world
            dist.all_gather_object(
                shards, {"error": error, "rows": local_rows, "best": local_best}
            )
            errors = [shard["error"] for shard in shards if shard["error"]]
            if errors:
                raise RuntimeError(f"distributed Quack tuning failed: {errors}")
            reported = [row for shard in shards for row in shard["rows"]]
            combined = dict(reported)
            if len(reported) != len(labels) or set(combined) != set(labels):
                raise RuntimeError("distributed tuning did not cover every candidate exactly once")
            rows = [(label, combined[label]) for label in labels]
            local_winners = [shard["best"] for shard in shards if shard["best"]]
            global_winner = min(local_winners, key=lambda row: (row[1], labels.index(row[0])))
            tuner.bench_time = time.monotonic() - started
            if rank == 0:
                if not _failed(value for _, value in rows) and native_allocator():
                    cache.put(
                        json.dumps({
                            "key": str(tuning_key),
                            "configs_timings": rows,
                            "tuning": {
                                "strategy": "distributed",
                                "world_size": world,
                                "warmup_ms": warmup,
                                "timed_ms": timed,
                                "inputs": "caller-synchronized activations",
                                "local_winners": [shard["best"] for shard in shards],
                                "global_winner": global_winner,
                            },
                        }),
                        filename, binary=False,
                    )
                print(
                    f"mlops quack: distributed search {name} completed "
                    f"{len(full_configs)} candidates in {tuner.bench_time:.1f}s",
                    flush=True,
                )
        by_label = dict(zip(labels, full_configs, strict=True))
        timings = {by_label[label]: value for label, value in rows}
        if _failed(timings.values()):
            raise RuntimeError(f"every distributed tuning candidate failed for {name}")
        winner = [str(builtins.min(timings, key=timings.get)) if rank == 0 else None]
        dist.broadcast_object_list(winner, src=0)
        tuner.configs_timings = timings
        tuner.cache[tuning_key] = by_label[winner[0]]
        if rank == 0 and tuner.bench_time:
            print(f"mlops quack: global winner {name}: {winner[0]}", flush=True)
    finally:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
