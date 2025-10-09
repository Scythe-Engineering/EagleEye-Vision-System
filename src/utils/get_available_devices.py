import glob
import os
import platform
import subprocess

import torch

from src.utils.colors import Colors


def get_cpu_name():
    try:
        if os.name == "nt":
            return platform.processor()
        elif os.name == "posix":
            result = subprocess.run(
                ["lscpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            for line in result.stdout.splitlines():
                if "Model name" in line:
                    return line.split(":")[1].strip()
        return "Unknown CPU"
    except Exception:
        return "Unknown CPU"


def get_gpu_devices() -> list[str]:
    """Get available GPU devices using PyTorch."""
    gpu_devices = []
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_devices.append(torch.cuda.get_device_name(i))
    return gpu_devices


def get_coral_tpu_devices() -> list[str]:
    """Get available Coral TPU devices."""
    coral_devices = []
    try:
        result = subprocess.run(
            ["lsusb"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        coral_count = 0
        for line in result.stdout.splitlines():
            if "Global Unichip Corp." in line:
                coral_devices.append(f"coral:{coral_count}")
                coral_count += 1
    except FileNotFoundError:
        pass
    return coral_devices


def get_memryx_tpu_devices() -> list[str]:
    """Get available Memryx TPU devices."""
    memryx_devices = []
    try:
        memryx_device_paths = glob.glob("/dev/memx*")
        for device_path in memryx_device_paths:
            device_name = device_path.split("/dev/memx")[1]
            # Only include main devices (exclude feature devices)
            if "_" not in device_name and "feature" not in device_name:
                memryx_devices.append(f"memx:{device_name}")
    except Exception:
        pass
    return memryx_devices


def get_tpu_devices() -> list[str]:
    """Get available TPU devices (Linux only)."""
    if os.name != "posix":
        return []

    tpu_devices = []
    tpu_devices.extend(get_coral_tpu_devices())
    tpu_devices.extend(get_memryx_tpu_devices())
    return tpu_devices


def get_available_devices():
    """Get all available compute devices."""
    devices = {
        "CPU": [get_cpu_name()],
        "GPU": get_gpu_devices(),
        "TPU": get_tpu_devices(),
    }
    return devices


if __name__ == "__main__":
    available_devices = get_available_devices()
    print(f"{Colors.CYAN}Available Devices:{Colors.RESET}", available_devices)
