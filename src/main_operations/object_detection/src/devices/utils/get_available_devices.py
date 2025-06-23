import os
import platform
import subprocess

import torch


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

def get_available_devices():
    devices = {"CPU": [get_cpu_name()], "GPU": [], "TPU": []}

    # Check for GPU availability using PyTorch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            devices["GPU"].append(torch.cuda.get_device_name(i))

    # Check for Coral TPU availability (Linux only)
    if os.name == "posix":
        try:
            result = subprocess.run(
                ["lsusb"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            coral_count = 0
            for line in result.stdout.splitlines():
                if "Global Unichip Corp." in line:
                    devices["TPU"].append(f"coral:{coral_count}")
                    coral_count += 1
        except FileNotFoundError:
            pass

    return devices

if __name__ == "__main__":
    available_devices = get_available_devices()
    print("Available Devices:", available_devices)

