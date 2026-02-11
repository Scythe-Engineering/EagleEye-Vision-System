from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.colors import Colors
from src.utils.logging.logger import Logger
from src.webui.web_server import EagleEyeInterface

if TYPE_CHECKING:
    from src.utils.camera_utils.camera_thread_manager import CameraThreadManager


def add_video_file_cameras(
    web_interface: EagleEyeInterface,
    camera_manager: "CameraThreadManager",
    logger: Logger,
) -> list[dict[str, str | int]]:
    """Add video file cameras to the system for testing and development.

    Cameras are started without calibration requirements - all calibration
    and rotation concerns are handled at the operation level.

    Args:
        web_interface (EagleEyeInterface): Web interface instance used to register
            video file cameras with the system for UI display and management.
        camera_manager (CameraThreadManager): Camera thread manager used to start
            camera capture threads for each video file.
        logger (Logger): Logger instance for outputting status messages about
            detected video files and registration results.

    Returns:
        list[dict[str, str | int]]: List of dictionaries containing camera information
            with "name", "index", and "bus_id" keys for each successfully registered
            video file camera. Returns an empty list if no video files are found in
            the sim_videos directory.

    Raises:
        No exceptions are raised; errors are logged and gracefully handled.
    """
    current_dir = Path(__file__).resolve().parent
    sim_videos_dir = current_dir.parent / "sim_videos"
    logger.log(
        f"{Colors.CYAN}Searching for sim videos in: {sim_videos_dir}{Colors.RESET}"
    )
    video_files = sorted(sim_videos_dir.glob("*.mp4"))
    logger.log(
        f"{Colors.CYAN}Found {len(video_files)} sim video files: {[vf.name for vf in video_files]}{Colors.RESET}"
    )

    known_cameras = []

    if not video_files:
        logger.log(
            f"{Colors.YELLOW}No video file cameras found in: {sim_videos_dir}{Colors.RESET}"
        )
        return known_cameras
    for video_file in video_files:
        camera_name = video_file.stem
        bus_id = camera_name
        web_interface.add_camera(camera_name, -1)
        started = camera_manager.start_camera_thread(
            camera_name,
            video_file_path=str(video_file),
        )
        if started:
            camera_manager.register_bus_id(bus_id, camera_name)
            known_cameras.append({"name": camera_name, "index": -1, "bus_id": bus_id})
            logger.log(
                f"{Colors.GREEN}Added video file camera: {camera_name}{Colors.RESET}"
            )
        else:
            web_interface.remove_camera(camera_name)
            logger.log(
                f"{Colors.RED}Failed to start video file camera: {camera_name}{Colors.RESET}"
            )
    return known_cameras
