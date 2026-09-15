"""Bounded Linux Vulkan prerequisite probe; no model or shader certification.

The LiteRT WebGPU delegate loads Vulkan lazily. Importing litert_lm and running
nvidia-smi can both succeed in a CUDA-only container which cannot create a GPU
engine. Keep these native loader/driver calls in a disposable process: malformed
or hung ICDs must not take down, or indefinitely block, the training launcher.
"""
from __future__ import annotations

import ctypes as ct
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

LOADER = "libvulkan.so.1"
PROBE_PREFIX = "A2UI_VULKAN_PROBE "
MIN_API_VERSION = (1 << 22) | (1 << 12)  # Vulkan 1.1
REPAIR_HINT = (
    "Install the Vulkan loader in the runtime image (Ubuntu/Debian: libvulkan1; "
    "vulkan-tools provides optional vulkaninfo diagnostics). The host NVIDIA "
    "driver and its Vulkan ICD must be available inside the container. Launch "
    "the job with NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics and only "
    "the scheduler-allocated GPUs; setting this inside an already-running "
    "container cannot mount missing driver libraries. Ask the platform admin "
    "to update/restart the container if needed. Run vulkaninfo --summary and "
    "the LiteRT --preflight in that same container before retrying. nvidia-smi "
    "alone proves CUDA/NVML visibility, not Vulkan/WebGPU. No CPU fallback is allowed."
)


class _ApplicationInfo(ct.Structure):
    _fields_ = [("sType", ct.c_uint32), ("pNext", ct.c_void_p),
                ("pApplicationName", ct.c_char_p), ("applicationVersion", ct.c_uint32),
                ("pEngineName", ct.c_char_p), ("engineVersion", ct.c_uint32), ("apiVersion", ct.c_uint32)]


class _InstanceCreateInfo(ct.Structure):
    _fields_ = [("sType", ct.c_uint32), ("pNext", ct.c_void_p), ("flags", ct.c_uint32),
                ("pApplicationInfo", ct.POINTER(_ApplicationInfo)), ("enabledLayerCount", ct.c_uint32),
                ("ppEnabledLayerNames", ct.c_void_p), ("enabledExtensionCount", ct.c_uint32),
                ("ppEnabledExtensionNames", ct.c_void_p)]


class _DeviceQueueCreateInfo(ct.Structure):
    _fields_ = [("sType", ct.c_uint32), ("pNext", ct.c_void_p), ("flags", ct.c_uint32),
                ("queueFamilyIndex", ct.c_uint32), ("queueCount", ct.c_uint32),
                ("pQueuePriorities", ct.POINTER(ct.c_float))]


class _DeviceCreateInfo(ct.Structure):
    _fields_ = [("sType", ct.c_uint32), ("pNext", ct.c_void_p), ("flags", ct.c_uint32),
                ("queueCreateInfoCount", ct.c_uint32), ("pQueueCreateInfos", ct.POINTER(_DeviceQueueCreateInfo)),
                ("enabledLayerCount", ct.c_uint32), ("ppEnabledLayerNames", ct.c_void_p),
                ("enabledExtensionCount", ct.c_uint32), ("ppEnabledExtensionNames", ct.c_void_p),
                ("pEnabledFeatures", ct.c_void_p)]


class _QueueFamilyProperties(ct.Structure):
    _fields_ = [("queueFlags", ct.c_uint32), ("queueCount", ct.c_uint32),
                ("timestampValidBits", ct.c_uint32), ("minImageTransferGranularity", ct.c_uint32 * 3)]


class _PropertiesPrefix(ct.Structure):
    # The remaining VkPhysicalDeviceProperties fields are deliberately opaque.
    # The complete core structure fits in the oversized, 8-byte-aligned 4096-byte
    # output buffer below. Never pass this short prefix as the native output.
    _fields_ = [("apiVersion", ct.c_uint32), ("driverVersion", ct.c_uint32),
                ("vendorID", ct.c_uint32), ("deviceID", ct.c_uint32), ("deviceType", ct.c_uint32),
                ("deviceName", ct.c_char * 256), ("pipelineCacheUUID", ct.c_ubyte * 16)]


def _bind(library: Any, name: str, result: Any, args: list[Any]) -> Any:
    function = getattr(library, name)
    function.restype, function.argtypes = result, args
    return function


def _check(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with VkResult={result}")


def _version(value: int) -> str:
    return f"{(value >> 22) & 127}.{(value >> 12) & 1023}.{value & 4095}"


def _device_probe(library: Any, physical: Any) -> dict[str, Any]:
    storage = (ct.c_uint64 * 512)()
    library.vkGetPhysicalDeviceProperties(physical, ct.byref(storage))
    properties = _PropertiesPrefix.from_buffer(storage)
    name = bytes(properties.deviceName).decode("utf-8", errors="replace")
    record: dict[str, Any] = {"name": name, "vendor_id": properties.vendorID,
                              "device_id": properties.deviceID, "device_type": properties.deviceType,
                              "api_version": _version(properties.apiVersion), "usable": False}
    if (properties.vendorID != 0x10DE or properties.deviceType not in (1, 2, 3)
            or any(word in name.lower() for word in ("llvmpipe", "lavapipe", "swiftshader", "software"))):
        return {**record, "reason": "Not a hardware NVIDIA GPU; software/CPU adapters are not accepted"}
    if properties.apiVersion < MIN_API_VERSION:
        return {**record, "reason": "GPU does not expose Vulkan 1.1 or newer"}
    count = ct.c_uint32()
    library.vkGetPhysicalDeviceQueueFamilyProperties(physical, ct.byref(count), None)
    if not 0 < count.value <= 256:
        return {**record, "reason": f"No valid queue inventory (count={count.value})"}
    queues = (_QueueFamilyProperties * count.value)()
    library.vkGetPhysicalDeviceQueueFamilyProperties(physical, ct.byref(count), queues)
    compute = next((index for index, queue in enumerate(queues[:count.value])
                    if queue.queueCount and queue.queueFlags & 2), None)
    if compute is None:
        return {**record, "reason": "No compute-capable Vulkan queue"}
    priority = ct.c_float(1.0)
    queue_info = _DeviceQueueCreateInfo(sType=2, queueFamilyIndex=compute, queueCount=1,
                                       pQueuePriorities=ct.pointer(priority))
    info = _DeviceCreateInfo(sType=3, queueCreateInfoCount=1, pQueueCreateInfos=ct.pointer(queue_info))
    device = ct.c_void_p()
    created_device = False
    try:
        _check(library.vkCreateDevice(physical, ct.byref(info), None, ct.byref(device)), "vkCreateDevice")
        if not device.value:
            raise RuntimeError("vkCreateDevice returned a null logical device")
        created_device = True
        queue = ct.c_void_p()
        library.vkGetDeviceQueue(device, compute, 0, ct.byref(queue))
        if not queue.value:
            raise RuntimeError("vkGetDeviceQueue returned a null compute queue")
        _check(library.vkQueueWaitIdle(queue), "vkQueueWaitIdle")
        return {**record, "usable": True, "compute_queue_family": compute,
                "logical_device_created": True, "queue_ready": True}
    except RuntimeError as exc:
        return {**record, "reason": str(exc)}
    finally:
        if created_device:
            library.vkDestroyDevice(device, None)


def _probe_in_process(*, library: Any = None) -> dict[str, Any]:
    """Private child operation. Call probe_vulkan_gpu from application code."""
    if library is None:
        try:
            library = ct.CDLL(LOADER)
        except OSError as exc:
            raise RuntimeError(f"Vulkan loader {LOADER} could not load: {exc}") from exc
    pointer = ct.c_void_p
    _bind(library, "vkCreateInstance", ct.c_int32, [pointer, pointer, ct.POINTER(pointer)])
    _bind(library, "vkDestroyInstance", None, [pointer, pointer])
    _bind(library, "vkEnumeratePhysicalDevices", ct.c_int32, [pointer, ct.POINTER(ct.c_uint32), pointer])
    _bind(library, "vkGetPhysicalDeviceProperties", None, [pointer, pointer])
    _bind(library, "vkGetPhysicalDeviceQueueFamilyProperties", None, [pointer, ct.POINTER(ct.c_uint32), pointer])
    _bind(library, "vkCreateDevice", ct.c_int32, [pointer, pointer, pointer, ct.POINTER(pointer)])
    _bind(library, "vkDestroyDevice", None, [pointer, pointer])
    _bind(library, "vkGetDeviceQueue", None, [pointer, ct.c_uint32, ct.c_uint32, ct.POINTER(pointer)])
    _bind(library, "vkQueueWaitIdle", ct.c_int32, [pointer])
    app = _ApplicationInfo(sType=0, pApplicationName=b"A2UI GPU prerequisites", apiVersion=MIN_API_VERSION)
    info = _InstanceCreateInfo(sType=1, pApplicationInfo=ct.pointer(app))
    instance = pointer()
    created_instance = False
    try:
        _check(library.vkCreateInstance(ct.byref(info), None, ct.byref(instance)), "vkCreateInstance")
        if not instance.value:
            raise RuntimeError("vkCreateInstance returned a null instance")
        created_instance = True
        count = ct.c_uint32()
        _check(library.vkEnumeratePhysicalDevices(instance, ct.byref(count), None), "vkEnumeratePhysicalDevices")
        if not 0 < count.value <= 256:
            raise RuntimeError(f"Vulkan reported no valid physical GPU inventory (count={count.value})")
        handles = (pointer * count.value)()
        _check(library.vkEnumeratePhysicalDevices(instance, ct.byref(count), handles), "vkEnumeratePhysicalDevices")
        devices = [_device_probe(library, handle) for handle in handles[:count.value]]
        usable = sum(device["usable"] is True for device in devices)
        report = {"status": "passed" if usable else "failed", "loader": LOADER,
                  "usable_device_count": usable, "devices": devices,
                  "model_kernel_tested": False, "webgpu_adapter_tested": False,
                  "gpu_affinity_verified": False, "probe_scope": "Vulkan hardware compute device and queue creation"}
        if not usable:
            report["error"] = "No usable hardware NVIDIA Vulkan compute device: " + "; ".join(
                f"{device['name']}: {device.get('reason', 'unavailable')}" for device in devices)
        return report
    finally:
        if created_instance:
            library.vkDestroyInstance(instance, None)


def probe_vulkan_gpu(*, timeout_seconds: float = 30) -> dict[str, Any]:
    """Run the loader/ICD/compute-device gate in the current runtime interpreter."""
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Vulkan probe timeout must be positive and finite")
    if sys.platform != "linux":
        raise RuntimeError("The NVIDIA Vulkan prerequisite probe requires Linux")
    command = [sys.executable, "-u", str(Path(__file__).absolute()), "--probe-child"]
    try:
        # subprocess.run kills and reaps this sole native child on timeout. It
        # inherits the stage's process group so outer cancellation also reaches it.
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=timeout_seconds, check=False, env=os.environ.copy())
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Vulkan native probe timed out after {timeout_seconds:g}s. {REPAIR_HINT}") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start the isolated Vulkan probe: {exc}. {REPAIR_HINT}") from exc
    reports = [line[len(PROBE_PREFIX):] for line in completed.stdout.splitlines() if line.startswith(PROBE_PREFIX)]
    if len(reports) != 1:
        diagnostic = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(f"Vulkan native probe exited {completed.returncode} without one valid report: "
                           f"{diagnostic}. {REPAIR_HINT}")
    try:
        report = json.loads(reports[0])
        if not isinstance(report, dict):
            raise TypeError("Report is not an object")
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Vulkan native probe returned an invalid report. {REPAIR_HINT}") from exc
    if completed.returncode or report.get("status") != "passed":
        raise RuntimeError(f"Vulkan GPU prerequisite failed: {report.get('error', 'unknown native error')}. {REPAIR_HINT}")
    devices = report.get("devices")
    if (not isinstance(devices, list) or not devices
            or any(not isinstance(device, dict) for device in devices)
            or type(report.get("usable_device_count")) is not int
            or report["usable_device_count"] <= 0
            or report["usable_device_count"] != sum(device.get("usable") is True for device in devices)
            or any(device.get("usable") is True and not (
                device.get("vendor_id") == 0x10DE and device.get("device_type") in (1, 2, 3)
                and device.get("logical_device_created") is True and device.get("queue_ready") is True
            ) for device in devices)):
        raise RuntimeError(f"Vulkan native probe lacks hardware compute-device evidence. {REPAIR_HINT}")
    return report


if __name__ == "__main__":
    if sys.argv[1:] != ["--probe-child"] or sys.platform != "linux":
        raise SystemExit("Internal Linux Vulkan probe; use run_litertlm_gpu.py --preflight")
    try:
        result = _probe_in_process()
    except Exception as exc:  # noqa: BLE001 - child serializes native/ABI failures before exiting nonzero
        result = {"status": "failed", "loader": LOADER, "error": f"{type(exc).__name__}: {exc}"}
    print(PROBE_PREFIX + json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)
