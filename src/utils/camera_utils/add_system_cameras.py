from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Set

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
    known_cameras: Set[str],
    logger: Logger | None = None,
) -> Set[str]:
    """Register every system camera with the camera manager.

    Cameras are started without calibration requirements - all calibration
    and rotation concerns are handled at the operation level.
    """

    detected_cameras = detect_cameras_with_names()
    if not detected_cameras:
        message = f"{Colors.YELLOW}{spacer}No system cameras detected.{Colors.RESET}"
        if logger:
            logger.log(message)
        else:
            print(message)
        return known_cameras

    camera_list = [camera_info["name"] for camera_info in detected_cameras.values()]
    message = (
        f"{Colors.CYAN}{spacer}Detected {len(camera_list)} system cameras: {camera_list}{Colors.RESET}"
    )
    if logger:
        logger.log(message)
    else:
        print(message)

    for index, camera_info in detected_cameras.items():
        camera_name = camera_info["name"]
        if camera_name in known_cameras:
            continue

        web_interface.add_camera(camera_name, index)
        try:
            camera_index = int(index)
        except ValueError:
            camera_index = None

        if camera_manager.start_camera_thread(
            camera_name, camera_index=camera_index
        ):
            known_cameras.add(camera_name)
        else:
            web_interface.remove_camera(camera_name)
            failure_message = (
                f"{Colors.RED}{spacer}Failed to start thread for system camera: {camera_name}{Colors.RESET}"
            )
            if logger:
                logger.log(failure_message)
            else:
                print(failure_message)

    return known_cameras
