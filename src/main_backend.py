import time
from typing import List

from src.utils.camera_utils.camera_thread_manager import CameraThreadManager
from src.utils.camera_utils.get_available_cameras import detect_cameras_with_names
from src.webui.web_server import EagleEyeInterface


def main() -> None:
    """Main function to initialize and start camera threads."""
    print("Initializing EagleEye backend...")

    web_interface = EagleEyeInterface()
    detected_cameras = detect_cameras_with_names()

    print(f"Detected {len(detected_cameras)} cameras: {list(detected_cameras.keys())}")

    if not detected_cameras:
        print("No cameras detected. Exiting...")
        return

    camera_manager = CameraThreadManager(web_interface)

    successful_cameras: List[str] = []
    failed_cameras: List[str] = []

    for camera_name, camera_index in detected_cameras.items():
        print(f"Starting camera thread for {camera_name} (index: {camera_index})")
        if camera_manager.start_camera_thread(camera_name, camera_index):
            successful_cameras.append(camera_name)
        else:
            failed_cameras.append(camera_name)

    print(f"Successfully started {len(successful_cameras)} camera threads")
    if failed_cameras:
        print(f"Failed to start {len(failed_cameras)} camera threads: {failed_cameras}")

    try:
        print("Camera threads running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        camera_manager.stop_all_cameras()
        print("Shutdown complete")


if __name__ == "__main__":
    main()
