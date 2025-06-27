from typing import Callable

import cv2
import imutils
import numpy as np

from src.utils.camera_utils.cameras.camera import Camera


class PhysicalCamera(Camera):
    """Concrete Camera that reads from a real hardware device via OpenCV."""

    def __init__(self, camera_data: dict, camera_intrinsics_path: str, log: Callable[[str], None] = print) -> None:
        """
        Args:
            camera_data: Must include 'camera_id' (int or str that OpenCV accepts).
            log: Logging function.
        """
        self.camera_id: int = camera_data["camera_id"]
        self.type = camera_data["camera_type"]
        super().__init__(camera_data, camera_intrinsics_path, log)

    def _start_camera(self) -> None:
        """Open the physical camera and apply settings."""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.fov[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.fov[1])
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error opening camera {self.camera_id}")

    def get_frame(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return imutils.rotate_bound(frame, self.frame_rotation)
