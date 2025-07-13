import subprocess
import platform

try:
    import torch
except ImportError:
    print("WARNING: torch not found")


def check_nvidia_gpu():
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_amd_gpu():
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                capture_output=True,
                text=True,
            )
        else:
            result = subprocess.run(["lspci"], capture_output=True, text=True)
        return "AMD" in result.stdout
    except subprocess.CalledProcessError:
        return False


def check_coral_tpu():
    try:
        result = subprocess.run(["lsusb"], capture_output=True, text=True)
        return ("Global Unichip Corp" in result.stdout) or (
            "Google Inc" in result.stdout
        )
    except subprocess.CalledProcessError:
        return False


def detect_hardware(log):
    """
    Detects the hardware available.

    Returns:
        tuple: A tuple containing two boolean values:
            - gpu_available (bool): True if a GPU is available, False otherwise.
            - tpu_available (bool): True if a TPU is available, False otherwise.
    """
    gpu_available = check_nvidia_gpu() or check_amd_gpu()
    tpu_available = check_coral_tpu()
    log(f"GPU Available: {gpu_available}")
    log(f"TPU Available: {tpu_available}")
    return gpu_available, tpu_available
