"""Cooperative tuner coordination, including cached and failing searches."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


class Config:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f"candidate-{self.index}"


def worker(rank, world, directory, scenario):
    import os

    os.environ["QUACK_CACHE_DIR"] = str(Path(directory) / "cache")
    dist.init_process_group(
        "gloo", init_method="file://" + str(Path(directory) / "rendezvous"),
        rank=rank, world_size=world, timeout=timedelta(seconds=60),
    )
    from mlops.quack_runtime import distributed

    configs = [Config(i) for i in range(7)]
    tuner = SimpleNamespace(
        fn=SimpleNamespace(__name__="test_search"), cache={}, bench_time=None
    )
    key = ("shape", rank if scenario == "mismatch" else 4096)
    visited = []

    def benchmark():
        if scenario == "failure" and rank == 1:
            raise ValueError("synthetic candidate failure")
        visited.extend(config.index for config in configs)
        tuner.configs_timings = {
            config: [float(abs(config.index - 5) + 1)] * 3 for config in configs
        }
        tuner.cache[key] = min(tuner.configs_timings, key=tuner.configs_timings.get)

    try:
        if scenario in ("failure", "mismatch"):
            with pytest.raises(RuntimeError, match="distributed Quack tuning"):
                distributed.check_disk_cache(tuner, key, configs, benchmark)
        else:
            distributed.check_disk_cache(tuner, key, configs, benchmark)
            assert tuner.cache[key].index == 5
            assert visited == list(range(rank, 7, world))
            assert [config.index for config in configs] == list(range(7))
            assert len(tuner.configs_timings) == 7
            previous = list(visited)
            tuner.cache.clear()
            distributed.check_disk_cache(tuner, key, configs, benchmark)
            assert tuner.cache[key].index == 5
            assert visited == previous and tuner.bench_time == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["success", "mismatch", "failure"])
def test_distributed_candidate_search(tmp_path, scenario):
    mp.spawn(worker, args=(3, str(tmp_path), scenario), nprocs=3, join=True)
