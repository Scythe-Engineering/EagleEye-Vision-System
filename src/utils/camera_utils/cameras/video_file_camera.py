import cv2
import numpy as np
from typing import Callable
from src.utils.camera_utils.cameras.camera import Camera
import imutils


class VideoFileCamera(Camera):
    """Concrete Camera that reads frames from a local video file."""

    def __init__(self, camera_name: str, camera_calibration_folder: str | None, video_file_path: str, log: Callable[[str], None] = print) -> None:
        """
        Args:
            camera_name: Name of the camera.
            camera_calibration_folder: Path to the camera calibration folder.
            video_file_path: Path to the video file.
            log: Logging function.
        """
        self.video_path = video_file_path
        super().__init__(camera_name, camera_calibration_folder, log)

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
        self.camera_ready = True
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
            self.current_frame_index = 0

        frame = self.frames[self.current_frame_index]
        self.current_frame_index += 1
        return frame

    def __del__(self):
        """Release the video capture object."""
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
