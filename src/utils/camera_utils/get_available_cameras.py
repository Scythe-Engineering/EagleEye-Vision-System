import platform
import subprocess
from subprocess import CalledProcessError

from src.utils.colors import Colors

# For Windows camera enumeration using DirectShow via pygrabber.
if platform.system() == "Windows":
    import comtypes

    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        raise ImportError(
            "pygrabber is required on Windows. Install with: pip install pygrabber"
        )


def _get_v4l2_device_output() -> str | None:
    """
    Gets the output from v4l2-ctl --list-devices command.

    Returns:
        The command output as a string, or None if the command fails.
    """
    try:
        output = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
        )
        return output
    except FileNotFoundError:
        print(
            f"{Colors.YELLOW}Warning: 'v4l2-ctl' is not installed. Please install v4l-utils package to get camera names on Linux.{Colors.RESET}"
        )
        return None
    except CalledProcessError:
        return None


def _is_device_name_line(line: str) -> bool:
    """Check if line contains a device name."""
    return line and not line.startswith(" ") and not line.startswith("\t")


def _should_skip_device(device_name: str) -> bool:
    """Check if device should be skipped."""
    return "pispbe" in device_name or "rpi-hevc-dec" in device_name


def _extract_camera_info(
    device_name: str, device_path: str
) -> tuple[str, dict[str, str]] | None:
    """Extract camera index and info from device name and path."""
    if not device_path.startswith("/dev/video"):
        return None

    camera_index = device_path.split("/dev/video")[1]
    clean_name = device_name.split(":")[0]
    bus_value = device_name.split(".")[-1][:-1]

    return camera_index, {"name": clean_name, "bus_value": bus_value}


def _parse_v4l2_output(output: str) -> dict[str, dict[str, str]]:
    """
    Parses v4l2-ctl output to extract camera device information.

    Args:
        output: The output string from v4l2-ctl --list-devices command.

    Returns:
        Dictionary mapping camera indices to dictionaries with camera names and bus values.
    """
    mapping = {}
    current_device_name = None
    device_name_line_index = 0

    for line_index, line in enumerate(output.splitlines()):
        if _is_device_name_line(line):
            current_device_name = line.rstrip(":").strip()
            device_name_line_index = line_index

            if _should_skip_device(current_device_name):
                current_device_name = None
        elif current_device_name and line_index == device_name_line_index + 1:
            camera_info = _extract_camera_info(current_device_name, line.strip())
            if camera_info:
                camera_index, info = camera_info
                mapping[camera_index] = info

    return mapping


def detect_linux_cameras() -> dict[str, dict[str, str]] | None:
    """
    Uses v4l2-ctl to get a mapping of video device files to camera names.

    Returns:
        Dictionary mapping camera indices to dictionaries with camera names and bus values.
        Returns None if v4l2-ctl is not available or fails.

    Note:
        v4l2-ctl must be installed (usually via v4l-utils package).
    """
    v4l2_output = _get_v4l2_device_output()
    if v4l2_output is None:
        return None

    return _parse_v4l2_output(v4l2_output)


def get_macos_camera_mapping() -> dict[str, str] | None:
    """
    Uses AVFoundation to get a list of available cameras on macOS.
    Returns:
        dict: Keys are device indices and values are camera names.
    """
    mapping = {}
    try:
        # Use system_profiler to list video devices
        output = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"], universal_newlines=True
        )
        current_name = None
        index = 0
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("Model ID:"):
                current_name = line.split(":")[1].strip()
                mapping[f"Camera {index}"] = current_name
                index += 1
    except CalledProcessError as e:
        print(
            f"{Colors.YELLOW}Warning: Unable to run 'system_profiler SPCameraDataType'. Camera names may not be available on macOS.{Colors.RESET}",
            e,
        )
    return mapping


def detect_windows_cameras() -> dict[str, dict[str, str]] | None:
    """
    Detect available cameras on Windows using DirectShow.

    Returns:
        Dictionary mapping camera indices to dictionaries with camera names and bus values.
    """
    try:
        comtypes.CoInitialize()
        graph = FilterGraph()
        device_names = graph.get_input_devices()

        mapping = {}
        for index, name in enumerate(device_names):
            mapping[str(index)] = {"name": name, "bus_value": str(index)}
        return mapping
    except Exception:
        return None


def detect_macos_cameras() -> dict[str, dict[str, str]] | None:
    """
    Detect available cameras on macOS using system_profiler.

    Returns:
        Dictionary mapping camera indices to dictionaries with camera names and bus values.
    """
    dev_name_mapping = get_macos_camera_mapping()
    if not dev_name_mapping:
        return None

    mapping = {}
    for index, name in enumerate(dev_name_mapping.values()):
        mapping[str(index)] = {"name": name, "bus_value": str(index)}
    return mapping


def detect_cameras_with_names() -> dict[str, dict[str, str]] | None:
    """
    Detect available cameras with their names and indices.

    Attempts to detect cameras using platform-specific methods to get
    meaningful names. Falls back to generic detection if platform-specific
    methods are not available.

    Args:
        max_tested: Maximum number of camera indices to test for generic detection.

    Returns:
        Dictionary mapping camera indices to dictionaries with camera names and bus values.
        Format: {"camera_index": {"name": "camera_name", "bus_value": "bus_value"}}
        Returns None if no cameras are detected or detection fails.
    """
    system = platform.system()
    if system == "Linux":
        return detect_linux_cameras()
    elif system == "Windows":
        return detect_windows_cameras()
    elif system == "Darwin":
        return detect_macos_cameras()


if __name__ == "__main__":
    detected = detect_cameras_with_names()
    if detected:
        print(f"{Colors.CYAN}Detected Cameras:{Colors.RESET}")
        for camera_index, camera_info in detected.items():
            print(
                f"{Colors.CYAN}Index: {camera_index}, Name: {camera_info['name']}, Bus Value: {camera_info.get('bus_value', 'N/A')}{Colors.RESET}"
            )
    else:
        print(f"{Colors.YELLOW}No cameras detected.{Colors.RESET}")
