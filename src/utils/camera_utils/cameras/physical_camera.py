from typing import Callable
import subprocess
import re

import cv2
import imutils
import numpy as np

from src.utils.camera_utils.cameras.camera import Camera
from src.utils.colors import Colors


class PhysicalCamera(Camera):
    """Concrete Camera that reads from a real hardware device via OpenCV."""

    def __init__(
        self,
        camera_name: str,
        camera_index: int,
        camera_calibration_folder: str | None,
        log: Callable[[str], None] = print,
    ) -> None:
        """
        Args:
            camera_name: Name of the camera.
            camera_index: Index of the camera.
            camera_calibration_folder: Path to the camera calibration folder.
            log: Logging function.
        """
        self.camera_index: int = camera_index
        self.achieved_fps: int = 30
        super().__init__(camera_name, camera_calibration_folder, log)

    def get_available_fps_for_resolution(self) -> list[int]:
        """
        Query available FPS for the configured resolution using v4l2-ctl.

        Returns:
            List of available FPS values in descending order, or empty list if unavailable.
        """
        device_path = f"/dev/video{self.camera_index}"
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", device_path, "--list-formats-ext"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            output = result.stdout
            available_fps = []

            resolution_pattern = (
                f"Size: Discrete {self.frame_width}x{self.frame_height}"
            )
            if resolution_pattern not in output:
                self.log(
                    f"{Colors.YELLOW}Resolution {self.frame_width}x{self.frame_height} not found in v4l2 formats{Colors.RESET}"
                )
                return []

            resolution_section = output.split(resolution_pattern)[1]
            next_resolution = resolution_section.split("Size: Discrete")
            if len(next_resolution) > 1:
                resolution_section = next_resolution[0]

            fps_pattern = r"Interval: Discrete [\d.]+s \(([\d.]+) fps\)"
            fps_matches = re.findall(fps_pattern, resolution_section)

            available_fps = sorted(
                [int(float(fps)) for fps in fps_matches], reverse=True
            )

            self.log(
                f"{Colors.CYAN}Available FPS for {self.frame_width}x{self.frame_height}: {available_fps}{Colors.RESET}"
            )
            return available_fps

        except FileNotFoundError:
            self.log(
                f"{Colors.YELLOW}v4l2-ctl not found, falling back to default FPS settings{Colors.RESET}"
            )
            return []
        except subprocess.TimeoutExpired:
            self.log(f"{Colors.YELLOW}v4l2-ctl query timed out{Colors.RESET}")
            return []
        except Exception as e:
            self.log(f"{Colors.RED}Error querying v4l2 formats: {e}{Colors.RESET}")
            return []

    def _start_camera(self) -> None:
        """Open the physical camera and apply settings."""
        self.cap = cv2.VideoCapture(int(self.camera_index))
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Error opening camera at index {self.camera_index} with name {self.name}"
            )

        # Set capture properties to reduce V4L2 timeouts
        # Set buffer size to reduce latency and avoid timeouts
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set frame width and height from extrinsics configuration
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Prefer MJPG codec for better performance and frame rate
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Get available FPS for the configured resolution
        available_fps = self.get_available_fps_for_resolution()

        if available_fps:
            target_fps = available_fps[0]
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.achieved_fps = int(actual_fps)
        else:
            self.achieved_fps = 15
            # Fallback to standard FPS values if v4l2-ctl is unavailable
            for target_fps in [120, 100, 90, 60, 30, 15]:
                self.cap.set(cv2.CAP_PROP_FPS, target_fps)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if actual_fps >= target_fps * 0.9:
                    self.achieved_fps = int(actual_fps)
                    break

        self.log(
            f"{Colors.GREEN}Camera {self.name}: Set resolution to {self.frame_width}x{self.frame_height} @ {self.achieved_fps} fps{Colors.RESET}"
        )

        # Enable autofocus if available
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        self.camera_ready = True

    def get_frame(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self.frame_rotation != 0:
            return imutils.rotate_bound(frame, self.frame_rotation)
        return frame

    def get_achieved_fps(self) -> int:
        return self.achieved_fps
