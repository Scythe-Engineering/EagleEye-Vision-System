import cv2
import numpy as np
from typing import Callable
from src.devices.utils.cameras.camera import Camera
import imutils


class VideoFileCamera(Camera):
    """Concrete Camera that reads frames from a local video file."""

    def __init__(self, camera_data: dict, log: Callable[[str], None]) -> None:
        """
        Args:
            camera_data: Must include 'video_path' (str, path to file)
                         and 'loop' (bool, whether to loop the video).
            log: Logging function.
        """
        self.video_path = camera_data["video_path"]
        self.loop = camera_data.get("loop", False)
        self.type = camera_data["camera_type"]
        super().__init__(camera_data, log)

        self.frames = self.load_frames()
        self.current_frame_index = 0

    def load_frames(self) -> list[np.ndarray]:
        """Load all frames from the video into a list."""
        print("Loading frames...")
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the start of the video
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(imutils.rotate_bound(frame, angle=self.frame_rotation))
        print("Frames loaded.")
        return frames

    def _start_camera(self) -> None:
        """Open the video file for reading."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error opening video file {self.video_path}")

    def get_frame(self) -> np.ndarray | None:
        """
        Read the next frame, rotate it, and return.
        Returns None when the video ends unless looping is enabled.
        """
        if self.current_frame_index >= len(self.frames):
            if self.loop:
                self.current_frame_index = 0
            else:
                return None

        frame = self.frames[self.current_frame_index]
        self.current_frame_index += 1
        return frame

    def __del__(self):
        """Release the video capture object."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
