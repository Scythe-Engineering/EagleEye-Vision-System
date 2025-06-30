from typing import Callable
from time import sleep

import cv2
import imutils
import numpy as np

from src.utils.camera_utils.cameras.camera import Camera


class PhysicalCamera(Camera):
    """Concrete Camera that reads from a real hardware device via OpenCV."""

    def __init__(self, camera_name: str, camera_index: int, camera_calibration_folder: str | None, log: Callable[[str], None] = print) -> None:
        """
        Args:
            camera_name: Name of the camera.
            camera_index: Index of the camera.
            camera_calibration_folder: Path to the camera calibration folder.
            log: Logging function.
        """
        self.camera_index: int = camera_index
        super().__init__(camera_name, camera_calibration_folder, log)

    def _start_camera(self) -> None:
        """Open the physical camera and apply settings."""
        self.cap = cv2.VideoCapture(int(self.camera_index))
        if not self.cap.isOpened():
            raise RuntimeError(f"Error opening camera at index {self.camera_index} with name {self.name}")

    def get_frame(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return imutils.rotate_bound(frame, self.frame_rotation)
