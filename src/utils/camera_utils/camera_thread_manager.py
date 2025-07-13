import threading
import time
import traceback
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from src.utils.camera_utils.cameras.physical_camera import PhysicalCamera
from src.utils.camera_utils.cameras.video_file_camera import VideoFileCamera
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
        self.camera_objects: Dict[str, Union[PhysicalCamera, VideoFileCamera]] = {}
        self.running_cameras: Dict[str, bool] = {}
        self.current_frames: Dict[str, Tuple[np.ndarray, float]] = {}
        self.start_time_ms = time.time() * 1000.0

    def camera_feed_worker(self, camera_name: str, camera: Union[PhysicalCamera, VideoFileCamera]) -> None:
        """
        Worker function that continuously captures frames and updates the web interface.

        Args:
            camera_name: The name of the camera.
            camera: The PhysicalCamera instance.
        """
        print(f"Starting camera feed worker for {camera_name}")
        
        frame_count = 0

        while self.running_cameras.get(camera_name, False):
            try:
                start_time = time.time()
                frame = camera.get_frame()
                frame_count += 1
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
                        
                        if frame_count % 10 == 0: # Update every 10 frames to reduce load on the web interface
                            self.web_interface.update_camera_frame(camera_name, frame_bytes)
                    else:
                        print(f"Failed to encode frame for {camera_name}")
                else:
                    print(f"Failed to get frame from {camera_name}")
                    time.sleep(0.1)

                time_to_sleep = 1 / 120 - (time.time() - start_time)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

            except Exception as camera_error:
                print(f"Error in camera feed worker for {camera_name}: {camera_error}")
                time.sleep(1)

        print(f"Camera feed worker for {camera_name} stopped")

    def start_camera_thread(self, camera_name: str, camera_calibration_folder: str | None, video_file_path: Optional[str] = None, camera_index: Optional[int] = None) -> bool:
        """
        Start a thread for a specific camera.

        Args:
            camera_name: The name of the camera.
            camera_calibration_folder: The path to the camera calibration folder.
            video_file_path: The path to the video file.

        Returns:
            True if the thread was started successfully, False otherwise.
        """
        try:
            if video_file_path:
                camera = VideoFileCamera(camera_name, camera_calibration_folder, video_file_path, print)
            else:
                if camera_index is None:
                    raise ValueError("Camera index is required for physical cameras")
                camera = PhysicalCamera(camera_name, camera_index, camera_calibration_folder, print)

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
            print(f"Full traceback: {traceback.format_exc()}")
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
    
    def get_all_camera_names(self) -> list[str]:
        """
        Get the names of all cameras.

        Returns:
            List of camera names.
        """
        return list(self.running_cameras.keys())
    
    def get_camera_ready(self, camera_name: str) -> bool:
        """
        Get the ready state of a specific camera.

        Args:
            camera_name: The name of the camera.
        """
        return self.camera_objects[camera_name].camera_ready
