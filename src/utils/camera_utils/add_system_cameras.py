from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.colors import Colors
from src.utils.camera_utils.get_available_cameras import detect_cameras_with_names
from src.utils.logging.logger import Logger
from src.webui.web_server import EagleEyeInterface

if TYPE_CHECKING:
    from src.utils.camera_utils.camera_thread_manager import CameraThreadManager

spacer = " " * 4


def add_system_cameras(
    web_interface: EagleEyeInterface,
    camera_manager: CameraThreadManager,
    logger: Logger,
) -> list[dict[str, str | int]]:
    """Register every system camera with the camera manager.

    Cameras are started without calibration requirements - all calibration
    and rotation concerns are handled at the operation level.

    Args:
        web_interface (EagleEyeInterface): Web interface instance used to register
            cameras with the system for UI display and management.
        camera_manager (CameraThreadManager): Camera thread manager used to start
            camera capture threads for each detected system camera.
        logger (Logger): Logger instance for outputting status messages about
            camera detection and registration progress.

    Returns:
        list[dict[str, str | int]]: List of dictionaries containing camera
            information with "name", "index", and "bus_id" keys for each
            successfully registered camera. Returns empty list if no system
            cameras are detected.

    Raises:
        No exceptions are raised; errors are logged and gracefully handled.
    """
    known_cameras = []

    detected_cameras = detect_cameras_with_names()
    if not detected_cameras:
        message = f"{Colors.YELLOW}{spacer}No system cameras detected.{Colors.RESET}"
        logger.log(message)
        return known_cameras

    camera_list = [camera_info["name"] for camera_info in detected_cameras.values()]
    message = f"{Colors.CYAN}{spacer}Detected {len(camera_list)} system cameras: {camera_list}{Colors.RESET}"
    logger.log(message)

    for index, camera_info in detected_cameras.items():
        camera_name = camera_info["name"]
        bus_id = camera_info.get("bus_id", "unknown")

        try:
            camera_index = int(index)
        except ValueError:
            camera_index = None

        web_interface.add_camera(camera_name, index)

        if camera_manager.start_camera_thread(camera_name, camera_index=camera_index):
            camera_manager.register_bus_id(bus_id, camera_name)
            known_cameras.append(
                {"name": camera_name, "index": camera_index, "bus_id": bus_id}
            )
        else:
            web_interface.remove_camera(camera_name)
            logger.log(
                f"{Colors.RED}{spacer}Failed to start thread for system camera: {camera_name}{Colors.RESET}"
            )

    return known_cameras
