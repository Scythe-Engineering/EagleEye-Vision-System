import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.utils.camera_utils.cameras.physical_camera import PhysicalCamera
from src.webui.web_server import EagleEyeInterface


class CameraThreadManager:
    """Manages camera threads and serves them to the web interface."""

    def __init__(self, web_interface: EagleEyeInterface) -> None:
        """
        Initialize the camera thread manager.

        Args:
            web_interface: The web interface to serve camera feeds to.
        """
        self.web_interface = web_interface
        self.camera_threads: Dict[str, threading.Thread] = {}
        self.camera_objects: Dict[str, PhysicalCamera] = {}
        self.running_cameras: Dict[str, bool] = {}
        self.current_frames: Dict[str, Tuple[np.ndarray, float]] = {}
        self.start_time_ms = time.time() * 1000.0

    def create_camera_config(self, camera_name: str, camera_index: int) -> dict:
        # TODO: Add correct camera config code
        """
        Create a camera configuration dictionary for the PhysicalCamera class.

        Args:
            camera_name: The name of the camera.
            camera_index: The index of the camera.

        Returns:
            Camera configuration dictionary.
        """
        return {
            "name": camera_name,
            "camera_id": camera_index,
            "camera_type": "physical",
            "fov": [640, 480],
            "camera_offset_pos": [0, 0, 0],
            "camera_pitch": 0,
            "camera_yaw": 0,
            "processing_device": "CPU",
            "frame_rotation": 0,
        }

    def camera_feed_worker(self, camera_name: str, camera: PhysicalCamera) -> None:
        """
        Worker function that continuously captures frames and updates the web interface.

        Args:
            camera_name: The name of the camera.
            camera: The PhysicalCamera instance.
        """
        print(f"Starting camera feed worker for {camera_name}")

        while self.running_cameras.get(camera_name, False):
            try:
                frame = camera.get_frame()
                if frame is not None:
                    current_time_ms = time.time() * 1000.0
                    timestamp_from_start = current_time_ms - self.start_time_ms

                    self.current_frames[camera_name] = (
                        frame.copy(),
                        timestamp_from_start,
                    )

                    success, encoded_frame = cv2.imencode(".jpg", frame)
                    if success:
                        frame_bytes = encoded_frame.tobytes()
                        self.web_interface.update_camera_frame(camera_name, frame_bytes)
                    else:
                        print(f"Failed to encode frame for {camera_name}")
                else:
                    print(f"Failed to get frame from {camera_name}")
                    time.sleep(0.1)

                time.sleep(1 / 30)  # Target 30 FPS

            except Exception as camera_error:
                print(f"Error in camera feed worker for {camera_name}: {camera_error}")
                time.sleep(1)

        print(f"Camera feed worker for {camera_name} stopped")

    def start_camera_thread(self, camera_name: str, camera_index: int) -> bool:
        """
        Start a thread for a specific camera.

        Args:
            camera_name: The name of the camera.
            camera_index: The index of the camera.

        Returns:
            True if the thread was started successfully, False otherwise.
        """
        try:
            camera_config = self.create_camera_config(camera_name, camera_index)
            camera = PhysicalCamera(camera_config, print)

            self.camera_objects[camera_name] = camera
            self.running_cameras[camera_name] = True

            camera_thread = threading.Thread(
                target=self.camera_feed_worker,
                args=(camera_name, camera),
                daemon=True,
                name=f"CameraThread-{camera_name}",
            )

            self.camera_threads[camera_name] = camera_thread
            camera_thread.start()

            print(
                f"Successfully started camera thread for {camera_name} (index: {camera_index})"
            )
            return True

        except Exception as start_error:
            print(f"Failed to start camera thread for {camera_name}: {start_error}")
            self.running_cameras[camera_name] = False
            return False

    def stop_camera_thread(self, camera_name: str) -> None:
        """
        Stop a specific camera thread.

        Args:
            camera_name: The name of the camera to stop.
        """
        if camera_name in self.running_cameras:
            print(f"Stopping camera thread for {camera_name}")
            self.running_cameras[camera_name] = False

            if camera_name in self.camera_threads:
                self.camera_threads[camera_name].join(timeout=5)
                del self.camera_threads[camera_name]

            if camera_name in self.camera_objects:
                del self.camera_objects[camera_name]

            if camera_name in self.current_frames:
                del self.current_frames[camera_name]

    def stop_all_cameras(self) -> None:
        """Stop all camera threads."""
        print("Stopping all camera threads...")
        camera_names = list(self.running_cameras.keys())
        for camera_name in camera_names:
            self.stop_camera_thread(camera_name)
        print("All camera threads stopped")

    def get_current_frame(self, camera_name: str) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the most current frame and timestamp for a specific camera.

        Args:
            camera_name: The name of the camera.

        Returns:
            Tuple of (frame, timestamp_ms_from_start) if available, None otherwise.
        """
        return self.current_frames.get(camera_name)

    def get_all_current_frames(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Get the most current frames and timestamps for all cameras.

        Returns:
            Dictionary mapping camera names to (frame, timestamp_ms_from_start) tuples.
        """
        return self.current_frames.copy()

    def get_start_time_ms(self) -> float:
        """
        Get the start time in milliseconds when the manager was initialized.

        Returns:
            Start time in milliseconds since epoch.
        """
        return self.start_time_ms
