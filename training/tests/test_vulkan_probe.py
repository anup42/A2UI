from __future__ import annotations

import ctypes as ct
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.eval import vulkan_probe as probe


class Function:
    def __init__(self, body):
        self.body = body

    def __call__(self, *args):
        return self.body(*args)


def _put(pointer, value, kind=ct.c_void_p):
    ct.cast(pointer, ct.POINTER(kind))[0] = value


def fake_library(*, devices=None, create_result=0, queue_flags=2, wait_result=0,
                 create_instance_result=0, enumerate_result=0, null_queue=False):
    devices = devices if devices is not None else [{"name": "NVIDIA H100", "vendor": 0x10DE, "type": 2}]
    destroyed, created = [], []

    def instance(info, allocator, out):
        assert ct.cast(info, ct.POINTER(probe._InstanceCreateInfo)).contents.pApplicationInfo.contents.apiVersion == probe.MIN_API_VERSION
        if not create_instance_result:
            _put(out, 100)
        return create_instance_result

    def enumerate_devices(instance, count, out):
        _put(count, len(devices), ct.c_uint32)
        if out is not None:
            for index in range(len(devices)):
                out[index] = index + 1
        return enumerate_result

    def properties(handle, output):
        fixture = devices[handle - 1]
        properties = ct.cast(output, ct.POINTER(probe._PropertiesPrefix)).contents
        properties.apiVersion = fixture.get("api", probe.MIN_API_VERSION)
        properties.vendorID = fixture["vendor"]
        properties.deviceType = fixture["type"]
        properties.deviceName = fixture["name"].encode()

    def queues(handle, count, out):
        _put(count, 1, ct.c_uint32)
        if out is not None:
            out[0].queueFlags, out[0].queueCount = queue_flags, 1

    def device(handle, info, allocator, out):
        created.append(handle)
        queue = ct.cast(info, ct.POINTER(probe._DeviceCreateInfo)).contents.pQueueCreateInfos.contents
        assert queue.queueCount == 1 and queue.pQueuePriorities[0] == 1.0
        if not create_result:
            _put(out, 200 + handle)
        return create_result

    def get_queue(device, family, index, out):
        _put(out, 0 if null_queue else 300)

    return SimpleNamespace(
        vkCreateInstance=Function(instance), vkDestroyInstance=Function(lambda handle, allocator: destroyed.append("instance")),
        vkEnumeratePhysicalDevices=Function(enumerate_devices), vkGetPhysicalDeviceProperties=Function(properties),
        vkGetPhysicalDeviceQueueFamilyProperties=Function(queues), vkCreateDevice=Function(device),
        vkDestroyDevice=Function(lambda handle, allocator: destroyed.append("device")),
        vkGetDeviceQueue=Function(get_queue), vkQueueWaitIdle=Function(lambda queue: wait_result),
        destroyed=destroyed, created=created,
    )


def test_real_ctypes_layout_probe_creates_and_cleans_hardware_compute_device():
    library = fake_library()
    result = probe._probe_in_process(library=library)
    assert result["status"] == "passed"
    assert result["usable_device_count"] == 1
    assert result["devices"][0]["logical_device_created"] is True
    assert result["devices"][0]["compute_queue_family"] == 0
    assert result["model_kernel_tested"] is False
    assert result["webgpu_adapter_tested"] is False
    assert result["gpu_affinity_verified"] is False
    assert library.destroyed == ["device", "instance"]


def test_loader_missing_preserves_exact_soname(monkeypatch):
    def missing(name):
        assert name == "libvulkan.so.1"
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(probe.ct, "CDLL", missing)
    with pytest.raises(RuntimeError, match="Vulkan loader libvulkan.so.1 could not load"):
        probe._probe_in_process()


@pytest.mark.parametrize("device", [
    {"name": "llvmpipe", "vendor": 0x10005, "type": 4},
    {"name": "SwiftShader", "vendor": 0x10DE, "type": 2},
    {"name": "NVIDIA CPU", "vendor": 0x10DE, "type": 4},
    {"name": "Other GPU", "vendor": 0x8086, "type": 2},
    {"name": "NVIDIA old API", "vendor": 0x10DE, "type": 2, "api": 1 << 22},
])
def test_software_non_nvidia_and_old_devices_do_not_pass(device):
    library = fake_library(devices=[device])
    result = probe._probe_in_process(library=library)
    assert result["status"] == "failed" and result["usable_device_count"] == 0
    assert not library.created
    assert library.destroyed == ["instance"]


def test_hardware_and_software_inventory_accepts_only_hardware():
    result = probe._probe_in_process(library=fake_library(devices=[
        {"name": "llvmpipe", "vendor": 0x10005, "type": 4},
        {"name": "NVIDIA H100", "vendor": 0x10DE, "type": 2},
    ]))
    assert result["status"] == "passed" and result["usable_device_count"] == 1
    assert result["devices"][0]["usable"] is False


@pytest.mark.parametrize(("kwargs", "error", "destroyed"), [
    ({"queue_flags": 1}, "No compute-capable", ["instance"]),
    ({"create_result": -3}, "vkCreateDevice failed", ["instance"]),
    ({"wait_result": -4}, "vkQueueWaitIdle failed", ["device", "instance"]),
    ({"null_queue": True}, "null compute queue", ["device", "instance"]),
])
def test_logical_compute_creation_failures_are_not_loader_success(kwargs, error, destroyed):
    library = fake_library(**kwargs)
    result = probe._probe_in_process(library=library)
    assert result["status"] == "failed" and error in result["error"]
    assert library.destroyed == destroyed


@pytest.mark.parametrize(("kwargs", "error", "destroyed"), [
    ({"devices": []}, "no valid physical GPU", ["instance"]),
    ({"create_instance_result": -9}, "vkCreateInstance failed", []),
    ({"enumerate_result": 5}, "vkEnumeratePhysicalDevices failed", ["instance"]),
])
def test_invalid_driver_or_inventory_fails_and_releases_instance(kwargs, error, destroyed):
    library = fake_library(**kwargs)
    with pytest.raises(RuntimeError, match=error):
        probe._probe_in_process(library=library)
    assert library.destroyed == destroyed


def _child(monkeypatch, *, report=None, returncode=0, stdout=None):
    monkeypatch.setattr(probe.sys, "platform", "linux")
    report = report if report is not None else probe._probe_in_process(library=fake_library())

    def run(command, **kwargs):
        assert command[0] == sys.executable and command[-1] == "--probe-child"
        assert kwargs["timeout"] == 30 and kwargs["check"] is False
        return SimpleNamespace(returncode=returncode, stdout=stdout if stdout is not None else
                               "driver noise\n" + probe.PROBE_PREFIX + json.dumps(report) + "\n", stderr="native failure")

    monkeypatch.setattr(probe.subprocess, "run", run)
    return report


def test_isolated_child_uses_runtime_python_and_retains_evidence(monkeypatch):
    expected = _child(monkeypatch)
    assert probe.probe_vulkan_gpu() == expected


def test_child_failure_has_admin_repair_instructions(monkeypatch):
    _child(monkeypatch, report={"status": "failed", "error": "libvulkan.so.1 missing"}, returncode=1)
    with pytest.raises(RuntimeError, match="libvulkan.so.1 missing") as error:
        probe.probe_vulkan_gpu()
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" in str(error.value)
    assert "already-running" in str(error.value) and "No CPU fallback" in str(error.value)


@pytest.mark.parametrize("stdout", ["", probe.PROBE_PREFIX + "null", probe.PROBE_PREFIX + "not json",
                                   probe.PROBE_PREFIX + "{}\n" + probe.PROBE_PREFIX + "{}"])
def test_invalid_native_report_fails(monkeypatch, stdout):
    _child(monkeypatch, stdout=stdout)
    with pytest.raises(RuntimeError, match="report"):
        probe.probe_vulkan_gpu()


@pytest.mark.parametrize("modification", [
    {"usable_device_count": True}, {"usable_device_count": 0}, {"devices": ["fake"]},
    {"devices": [{"usable": True, "vendor_id": 0x10005, "device_type": 4}]},
])
def test_unproven_hardware_report_is_rejected(monkeypatch, modification):
    report = probe._probe_in_process(library=fake_library())
    report.update(modification)
    _child(monkeypatch, report=report)
    with pytest.raises(RuntimeError, match="lacks hardware compute-device evidence"):
        probe.probe_vulkan_gpu()


def test_native_hang_is_bounded(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "linux")

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(probe.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out after 30s"):
        probe.probe_vulkan_gpu()


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_invalid_timeout_rejected(timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        probe.probe_vulkan_gpu(timeout_seconds=timeout)
