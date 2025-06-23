import threading
from typing import Callable
from urllib.request import urlopen

import cv2
import imutils
import numpy as np

from src.constants.constants import constants
from src.devices.utils.cameras.camera import Camera

NetworkTableConstants = constants["NetworkTableConstants"]


def extract_and_decode_jpeg(buffer: bytearray) -> tuple[np.ndarray | None, bytearray]:
    a = buffer.find(b"\xff\xd8")
    b = buffer.find(b"\xff\xd9")
    if a != -1 and b != -1:
        jpg = buffer[a : b + 2]
        buffer = buffer[b + 2 :]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame, buffer
    return None, buffer

def sim_frame_reader(
    url: str, latest_frame_method: Callable[[np.ndarray], None]
) -> None:
    """
    Continuously read JPEG frames from an HTTP MJPEG stream
    and hand them to the provided callback.
    """
    stream = urlopen(url)
    buffer = bytearray()
    while True:
        try:
            while True:
                # prevent unbounded growth
                if len(buffer) > 100_000:
                    buffer = buffer[-100_000:]
                buffer += stream.read(4096)
                frame, buffer = extract_and_decode_jpeg(buffer)
                if frame is not None:
                    latest_frame_method(frame)
                    break
        except Exception as e:
            print("Exception in sim_frame_reader:", e)
            break


class SimCamera(Camera):
    """Concrete Camera that pulls frames from an MJPEG HTTP stream."""

    def __init__(self, camera_data: dict, log: Callable[[str], None]) -> None:
        """
        Args:
            camera_data: Must include 'camera_id' (the stream endpoint).
            log: Logging function.
        """
        self.camera_id: str = camera_data["camera_id"]
        self.latest_frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()
        self.type = camera_data["camera_type"]
        super().__init__(camera_data, log)

    def _start_camera(self) -> None:
        """Spin up a background thread to fetch frames over HTTP."""
        url = (
            f"http://{NetworkTableConstants.server_address}:1181/"
            f"{self.camera_id}?fps=60"
        )
        self.log(f"Using simulation stream at {url}")
        threading.Thread(
            target=sim_frame_reader, args=(url, self._set_frame), daemon=True
        ).start()

    def get_frame(self) -> np.ndarray | None:
        """Return the most recent frame (rotated), or None if not yet available."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return imutils.rotate_bound(self.latest_frame, self.frame_rotation)

    def _set_frame(self, frame: np.ndarray) -> None:
        """Internal: thread-safe update of the latest frame."""
        with self.frame_lock:
            self.latest_frame = frame
