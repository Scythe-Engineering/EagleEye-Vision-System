import cv2
import imutils
import numpy as np
from typing import Callable
from src.devices.utils.cameras.camera import Camera


class PhysicalCamera(Camera):
    """Concrete Camera that reads from a real hardware device via OpenCV."""

    def __init__(self, camera_data: dict, log: Callable[[str], None]) -> None:
        """
        Args:
            camera_data: Must include 'camera_id' (int or str that OpenCV accepts).
            log: Logging function.
        """
        self.camera_id: int = camera_data["camera_id"]
        self.type = camera_data["camera_type"]
        super().__init__(camera_data, log)

    def _start_camera(self) -> None:
        """Open the physical camera and apply settings."""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.fov[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.fov[1])
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error opening camera {self.camera_id}")

    def get_frame(self) -> np.ndarray | None:
        """Grab the next frame and apply rotation."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return imutils.rotate_bound(frame, self.frame_rotation)
