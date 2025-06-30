import os
from pathlib import Path
from typing import Set

from src.utils.camera_utils.camera_thread_manager import CameraThreadManager
from src.webui.web_server import EagleEyeInterface
from src.utils.camera_utils.get_available_cameras import detect_cameras_with_names

current_dir = Path(__file__).parent
spacer = " " * 4

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
    
    if detected_cameras is None:
        return known_cameras
    
    new_cameras = {}

    for index, camera_info in detected_cameras.items():
        camera_name = camera_info["name"]
        camera_info["index"] = index
        if camera_name not in known_cameras:
            new_cameras[camera_name] = camera_info
            
    if new_cameras:
        print("Camera Detection Info:")
        print(f"{spacer}Found {len(new_cameras)} new cameras: {list(new_cameras.keys())}")

        for camera_name, camera_info in new_cameras.items():
            calibration_folder = os.path.join(current_dir, "camera_calibrations", f"{camera_info['bus_value']}")
            
            if not os.path.exists(calibration_folder):
                calibration_folder = None
            
            web_interface.add_camera(camera_name, camera_info["index"])
            print(
                f"{spacer}Added new camera to web interface: {camera_name} (index: {camera_info['index']})"
            )

            if camera_manager.start_camera_thread(camera_name, calibration_folder, camera_index=camera_info["index"]):
                known_cameras.add(camera_name)
                print(f"{spacer}Successfully started thread for new camera: {camera_name}")
            else:
                web_interface.remove_camera(camera_name)
                print(f"{spacer}Failed to start thread for new camera: {camera_name}")

    return known_cameras