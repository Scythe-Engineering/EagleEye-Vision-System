import os
from pathlib import Path
import time
from typing import Set

from src.utils.camera_utils.camera_thread_manager import CameraThreadManager
from src.utils.camera_utils.get_available_cameras import detect_cameras_with_names
from src.webui.web_server import EagleEyeInterface

current_dir = Path(__file__).parent


def check_and_add_new_cameras(
    web_interface: EagleEyeInterface,
    camera_manager: CameraThreadManager,
    known_cameras: Set[str],
) -> Set[str]:
    """
    Check for new cameras and add them to the system.

    Args:
        web_interface: The web interface instance.
        camera_manager: The camera thread manager instance.
        known_cameras: Set of camera names already known to the system.

    Returns:
        Updated set of known camera names.
    """
    detected_cameras = detect_cameras_with_names()
    new_cameras = {}

    for camera_name, camera_index in detected_cameras.items():
        if camera_name not in known_cameras:
            new_cameras[camera_name] = camera_index

    if new_cameras:
        print(f"Found {len(new_cameras)} new cameras: {list(new_cameras.keys())}")

        for camera_name, camera_index in new_cameras.items():
            web_interface.add_camera(camera_name, camera_index)
            print(
                f"Added new camera to web interface: {camera_name} (index: {camera_index})"
            )

            if camera_manager.start_camera_thread(camera_name, camera_index):
                known_cameras.add(camera_name)
                print(f"Successfully started thread for new camera: {camera_name}")
            else:
                web_interface.remove_camera(camera_name)
                print(f"Failed to start thread for new camera: {camera_name}")

    return known_cameras


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
        camera_index = len(known_cameras)
        web_interface.add_camera(camera_name, camera_index)
        camera_manager.start_camera_thread(camera_name, camera_index)
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
