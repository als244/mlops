"""Driver lifecycle controls around the actual pinned HardwareInfo queries."""

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest

from mlops.quack_runtime import _occupancy as occupancy


@pytest.fixture
def driver_control(monkeypatch, tmp_path):
    pytest.importorskip("cutlass")
    from cuda.bindings import driver
    from cutlass import cute

    environment = {"target": "sm_90a", "source": "test-source"}
    monkeypatch.setattr(occupancy, "_environment", lambda device: environment)
    state = {
        "context": 50,
        "device": 0,
        "count": 1,
        "capacity": 80,
        "loads": [],
        "unloads": [],
        "queries": [],
    }
    success = driver.CUresult.CUDA_SUCCESS

    def result(name, function):
        monkeypatch.setattr(driver, name, lambda *args: (success, function(*args)))

    result("cuDeviceGetCount", lambda: 2)
    result("cuDeviceGet", lambda device: device)
    result("cuCtxGetCurrent", lambda: state["context"])
    result("cuCtxGetDevice", lambda: state["device"])
    result("cuDriverGetVersion", lambda: 13020)
    result(
        "cuLibraryLoadData",
        lambda *args: state["loads"].append(args[0]) or len(state["loads"]),
    )
    result("cuLibraryGetKernelCount", lambda library: state["count"])
    result("cuLibraryEnumerateKernels", lambda count, library: [library + 100])
    result("cuKernelGetFunction", lambda kernel: kernel + 100)
    result("cuDeviceGetAttribute", lambda attribute, device: 227 * 1024)
    result("cuOccupancyAvailableDynamicSMemPerBlock", lambda *args: 220 * 1024)
    result("cuOccupancyMaxActiveBlocksPerMultiprocessor", lambda *args: 1)
    monkeypatch.setattr(driver, "cuFuncSetAttribute", lambda *args: (success,))

    def clusters(kernel, config):
        size = config.attrs[0].value.clusterDim.x
        state["queries"].append(
            (kernel, size, int(config.hStream), config.sharedMemBytes)
        )
        return state["capacity"] // size

    result("cuOccupancyMaxActiveClusters", clusters)
    monkeypatch.setattr(
        driver,
        "cuLibraryUnload",
        lambda library: (state["unloads"].append(library) or success,),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary helper tried to compile")

    monkeypatch.setattr(cute, "compile", forbidden)
    helper = occupancy.OccupancyHelper(tmp_path, device="cuda:0")
    data = b"driver-fixture"
    manifest = {
        "schema": occupancy._SCHEMA,
        "source": occupancy._SOURCE,
        "environment": copy.deepcopy(environment),
        "sha256": occupancy._digest(data),
        "bytes": len(data),
    }
    return helper, manifest, lambda _: data, state


def test_actual_query_uses_live_capacity_cluster_and_stream(driver_control):
    from cuda.bindings import driver

    helper, manifest, read, state = driver_control
    loaded = helper.load(manifest, read)
    hardware = occupancy._hardware_info()(0)
    with loaded.activate():
        assert hardware.get_max_active_clusters(1) == 80
        state["capacity"] = 96
        assert hardware.get_max_active_clusters(2, driver.CUstream(42)) == 48
        with pytest.raises(RuntimeError, match="active"):
            loaded.close()
    assert state["queries"] == [(201, 1, 0, 220 * 1024), (201, 2, 42, 220 * 1024)]
    assert len(state["loads"]) == 1
    loaded.close()
    loaded.close()
    assert state["unloads"] == [1]
    with pytest.raises(RuntimeError, match="closed"), loaded.activate():
        pass


def test_nested_and_preexisting_thread_scopes_restore_ordinary_behavior(driver_control):
    helper, manifest, read, state = driver_control
    left, right = helper.load(manifest, read), helper.load(manifest, read)
    hardware = occupancy._hardware_info()(0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(lambda: 1).result() == 1
        with left.activate():
            assert hardware._get_device_function() == 201
            with pytest.raises(AssertionError, match="tried to compile"):
                pool.submit(hardware._get_device_function).result()
            with pytest.raises(RuntimeError, match="inner failure"), right.activate():
                assert hardware._get_device_function() == 202
                raise RuntimeError("inner failure")
            assert hardware._get_device_function() == 201
    with pytest.raises(AssertionError, match="tried to compile"):
        hardware._get_device_function()
    left.close()
    right.close()
    assert state["unloads"] == [1, 2]


def test_context_mismatch_and_partial_load_release(driver_control):
    helper, manifest, read, state = driver_control
    state["device"] = 1
    with pytest.raises(ValueError, match="not current"):
        helper.load(manifest, read)
    assert not state["loads"]
    state["device"] = 0
    state["count"] = 2
    with pytest.raises(ValueError, match="exactly one kernel"):
        helper.load(manifest, read)
    assert state["unloads"] == [1]
    state["count"] = 1
    loaded = helper.load(manifest, read)
    hardware = occupancy._hardware_info()(0)
    with loaded.activate():
        state["context"] = 51
        with pytest.raises(ValueError, match="another device/context"):
            hardware._get_device_function()
    loaded.close()
    assert state["unloads"] == [1, 2]


def test_manifest_and_corruption_refuse_before_driver_loading(driver_control):
    helper, manifest, read, state = driver_control
    broken = copy.deepcopy(manifest)
    broken["environment"]["target"] = "sm_120a"
    with pytest.raises(ValueError, match="incompatible"):
        helper.load(broken, read)
    with pytest.raises(ValueError, match="corrupt"):
        helper.load(manifest, lambda _: b"corrupt")
    assert not state["loads"]
