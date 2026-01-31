from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Set

from src.utils.colors import Colors
from src.utils.logging.logger import Logger
from src.webui.web_server import EagleEyeInterface

if TYPE_CHECKING:
    from src.utils.camera_utils.camera_thread_manager import CameraThreadManager


def add_video_file_cameras(
    web_interface: EagleEyeInterface,
    camera_manager: "CameraThreadManager",
    known_cameras: Set[str],
    logger: Logger,
) -> Set[str]:
    """
    Add video file cameras to the system. (Mostly for testing and development purposes)
    """
    current_dir = Path(__file__).parent
    video_folder = os.path.join(current_dir, "sim_videos")
    video_files = list(Path(video_folder).glob("*.mp4"))
    for video_file in video_files:
        camera_name = video_file.stem
        web_interface.add_camera(camera_name, -1)
        camera_manager.start_camera_thread(
            camera_name,
            os.path.join(
                current_dir,
                "camera_calibrations",
                "sim_camera",
            ),
            video_file_path=str(video_file),
        )
        known_cameras.add(camera_name)
        logger.log(
            f"{Colors.GREEN}Added video file camera: {camera_name}{Colors.RESET}"
        )
    return known_cameras
