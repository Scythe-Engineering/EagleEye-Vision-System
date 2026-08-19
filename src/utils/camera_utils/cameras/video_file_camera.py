import time
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.camera_utils.cameras.camera import Camera
from src.utils.camera_utils.cameras.captured_frame import CapturedFrame
from src.utils.colors import Colors


class VideoFileCamera(Camera):
    """Concrete Camera that reads frames from a local video file."""

    def __init__(
        self,
        camera_name: str,
        video_file_path: str,
        log: Callable[[str], None] = print,
    ) -> None:
        """Initialize the video file camera.

        Args:
            camera_name: Name of the camera.
            video_file_path: Path to the video file.
            log: Logging function.
        """
        self.video_path = video_file_path
        super().__init__(camera_name, log)

        self.frames = self.load_frames()
        self.current_frame_index = 0

    def load_frames(self) -> list[np.ndarray]:
        """Load all frames from the video into a list without rotation.

        Returns:
            List of raw frames from the video file.
        """
        self.log(
            f"{Colors.CYAN}Loading frames (will init after frames are loaded into ram)...{Colors.RESET}"
        )
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in tqdm(range(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        self.log(f"{Colors.GREEN}Frames loaded.{Colors.RESET}")
        self.camera_ready = True
        return frames

    def _start_camera(self) -> None:
        """Open the video file for reading."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error opening video file {self.video_path}")

    def get_frame(self) -> CapturedFrame | None:
        """Read the next raw frame without rotation.

        Replayed frames have no meaningful capture time, so they are stamped
        when handed out.

        Returns:
            The next frame with its delivery timestamp, or None when the file
            holds no frames. Playback loops back to the start.
        """
        if not self.frames:
            return None

        if self.current_frame_index >= len(self.frames):
            self.current_frame_index = 0

        frame = self.frames[self.current_frame_index]
        self.current_frame_index += 1
        return CapturedFrame(image=frame, capture_monotonic_ns=time.monotonic_ns())

    def get_frame_index(self) -> int:
        """Return the current frame index."""
        return self.current_frame_index

    def get_achieved_fps(self) -> int:
        """Get the FPS that the video file is set to play at."""
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def close(self) -> None:
        """Release the video capture object."""
        if getattr(self, "cap", None) is not None and self.cap.isOpened():
            self.cap.release()

    def __del__(self) -> None:
        """Release the video capture object."""
        self.close()
