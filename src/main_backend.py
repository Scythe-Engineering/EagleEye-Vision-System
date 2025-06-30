import os
from pathlib import Path
import time
from typing import Set

from src.utils.camera_utils.camera_thread_manager import CameraThreadManager
from src.utils.camera_utils.check_and_add_new_cameras import check_and_add_new_cameras
from src.webui.web_server import EagleEyeInterface

current_dir = Path(__file__).parent
spacer = " " * 4


def add_video_file_cameras(
    web_interface: EagleEyeInterface,
    camera_manager: CameraThreadManager,
    known_cameras: Set[str],
) -> None:
    """
    Add video file cameras to the system.
    """
    # Add a video file cameras
    video_folder = os.path.join(current_dir, "utils", "sim_videos")
    video_files = list(Path(video_folder).glob("*.mp4"))
    for video_file in video_files:
        camera_name = video_file.stem
        web_interface.add_camera(camera_name, -1)
        camera_manager.start_camera_thread(camera_name, os.path.join(current_dir, "utils", "camera_utils", "camera_calibrations", "sim_camera"), str(video_file))
        known_cameras.add(camera_name)
        print(f"Added video file camera: {camera_name}")


def main() -> None:
    """Main function to initialize and continuously monitor for cameras."""
    print("Initializing EagleEye backend...")

    web_interface = EagleEyeInterface()
    camera_manager = CameraThreadManager(web_interface)
    known_cameras: Set[str] = set()

    # Initial camera detection
    print("Performing initial camera detection...")
    known_cameras = check_and_add_new_cameras(
        web_interface, camera_manager, known_cameras
    )

    add_video_file_cameras(web_interface, camera_manager, known_cameras)

    if not known_cameras:
        print(
            "No cameras detected initially. Will continue checking for new cameras..."
        )
    else:
        print(f"Initially started {len(known_cameras)} cameras: {list(known_cameras)}")

    try:
        print(
            "Camera monitoring active. Checking for new cameras every 5 seconds. Press Ctrl+C to stop..."
        )

        while True:
            known_cameras = check_and_add_new_cameras(
                web_interface, camera_manager, known_cameras
            )

            time.sleep(5)

    except KeyboardInterrupt:
        print("\nShutting down...")
        camera_manager.stop_all_cameras()
        print("Shutdown complete")


if __name__ == "__main__":
    main()
