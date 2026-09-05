from typing import Callable
import re
import subprocess
import time

import cv2

from src.utils.camera_utils.cameras.camera import Camera
from src.utils.camera_utils.cameras.captured_frame import CapturedFrame
from src.utils.camera_utils.cameras.v4l2_capture import (
    CID_FOCUS_AUTO,
    V4l2Capture,
    v4l2_is_supported,
)
from src.utils.colors import Colors

FALLBACK_FPS_CANDIDATES = (120, 100, 90, 60, 30, 15)
DEFAULT_FPS = 15


class OpenCvCapture:
    """OpenCV capture fallback for platforms without V4L2.

    Frames are stamped when ``read()`` returns, so the timestamp includes
    exposure, transfer, and decode. This is the accuracy the whole pipeline had
    before the V4L2 backend existed; it remains acceptable for development on
    non-Linux hosts, where nothing is being localized.
    """

    def __init__(self, camera_index: int, frame_width: int, frame_height: int) -> None:
        """Open the camera and request MJPEG at the given resolution.

        Args:
            camera_index: OpenCV camera device index.
            frame_width: Requested frame width in pixels.
            frame_height: Requested frame height in pixels.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self.capture = cv2.VideoCapture(int(camera_index))
        if not self.capture.isOpened():
            raise RuntimeError(f"Error opening camera at index {camera_index}")

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore
        self.frame_rate = 0
        self.timestamp_source = "delivery"

    def set_frame_rate(self, frames_per_second: int) -> int:
        """Request a capture rate.

        Args:
            frames_per_second: Desired capture rate.

        Returns:
            The rate reported by OpenCV, or zero when unavailable.
        """
        self.capture.set(cv2.CAP_PROP_FPS, frames_per_second)
        self.frame_rate = int(self.capture.get(cv2.CAP_PROP_FPS))
        return self.frame_rate

    def start(self) -> None:
        """OpenCV starts capture when the device is opened."""

    def set_control(self, control_id: int, value: int) -> bool:
        """Apply an equivalent OpenCV camera control when supported.

        Args:
            control_id: V4L2 control identifier.
            value: Control value to apply.

        Returns:
            Whether OpenCV accepted the control.
        """
        if control_id != CID_FOCUS_AUTO:
            return False
        return bool(self.capture.set(cv2.CAP_PROP_AUTOFOCUS, value))

    def read(self, timeout_s: float = 1.0) -> CapturedFrame | None:
        """Read one frame, stamping it at delivery time.

        Args:
            timeout_s: Ignored because OpenCV provides no read timeout.

        Returns:
            The delivered frame, or ``None`` on read failure.
        """
        del timeout_s  # cv2.VideoCapture.read() has no timeout control.
        success, frame = self.capture.read()
        if not success or frame is None:
            return None
        return CapturedFrame(image=frame, capture_monotonic_ns=time.monotonic_ns())

    def close(self) -> None:
        """Release the underlying capture."""
        if self.capture is not None:
            self.capture.release()


class PhysicalCamera(Camera):
    """Concrete Camera that reads from a real hardware device.

    On Linux this streams through V4L2 directly so each frame carries the
    kernel's capture timestamp. Elsewhere it falls back to OpenCV.
    """

    def __init__(
        self,
        camera_name: str,
        camera_index: int,
        frame_width: int = 1280,
        frame_height: int = 720,
        log: Callable[[str], None] = print,
    ) -> None:
        """Initialize the physical camera.

        Args:
            camera_name: Name of the camera.
            camera_index: Index of the camera device.
            frame_width: Desired frame width in pixels. Defaults to 1280.
            frame_height: Desired frame height in pixels. Defaults to 720.
            log: Logging function.
        """
        self.camera_index: int = camera_index
        self.frame_width: int = frame_width
        self.frame_height: int = frame_height
        self.achieved_fps: int = 30
        self.backend: V4l2Capture | OpenCvCapture | None = None
        super().__init__(camera_name, log)

    @property
    def device_path(self) -> str:
        """V4L2 device node for this camera index."""
        return f"/dev/video{self.camera_index}"

    def get_available_fps_for_resolution(self) -> list[int]:
        """Query available FPS for the configured resolution using v4l2-ctl.

        Returns:
            List of available FPS values in descending order, or empty list if unavailable.
        """
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", self.device_path, "--list-formats-ext"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            output = result.stdout

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

    def _open_backend(self) -> V4l2Capture | OpenCvCapture:
        """Create the capture backend appropriate for this platform.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        if v4l2_is_supported():
            return V4l2Capture(
                self.device_path,
                self.frame_width,
                self.frame_height,
                log=self.log,
            )
        return OpenCvCapture(self.camera_index, self.frame_width, self.frame_height)

    def _negotiate_frame_rate(
        self, backend: V4l2Capture | OpenCvCapture, available_fps: list[int]
    ) -> int:
        """Pick the fastest rate the device will accept for this resolution."""
        if available_fps:
            achieved_fps = backend.set_frame_rate(available_fps[0])
            if achieved_fps > 0:
                return achieved_fps

        for candidate_fps in FALLBACK_FPS_CANDIDATES:
            achieved_fps = backend.set_frame_rate(candidate_fps)
            if achieved_fps >= candidate_fps * 0.9:
                return achieved_fps
        return DEFAULT_FPS

    def _start_camera(self) -> None:
        """Open the physical camera and apply settings.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        # Discover capabilities before opening the capture handle; avoid a second
        # device open by v4l2-ctl while configuring or streaming the camera.
        available_fps = self.get_available_fps_for_resolution()
        try:
            backend = self._open_backend()
        except Exception as error:
            raise RuntimeError(
                f"Error opening camera at index {self.camera_index} "
                f"with name {self.name}: {error}"
            ) from error

        self.backend = backend
        self.achieved_fps = self._negotiate_frame_rate(backend, available_fps)
        backend.start()
        backend.set_control(CID_FOCUS_AUTO, 1)

        self.log(
            f"{Colors.GREEN}Camera {self.name}: {self.frame_width}x{self.frame_height} "
            f"@ {self.achieved_fps} fps via {type(backend).__name__}{Colors.RESET}"
        )

        self.camera_ready = True

    def get_frame(self) -> CapturedFrame | None:
        """Read the newest frame and its capture time, without rotation.

        Returns:
            The frame with its capture timestamp, or None on read failure.
        """
        if self.backend is None:
            return None
        return self.backend.read()

    def get_achieved_fps(self) -> int:
        """Get the FPS that the camera is operating at.

        Returns:
            int: The achieved frames per second of the camera, representing the
                actual capture rate.
        """
        return self.achieved_fps

    def close(self) -> None:
        """Stop streaming and release the device."""
        if self.backend is not None:
            self.backend.close()
            self.backend = None
        self.camera_ready = False
