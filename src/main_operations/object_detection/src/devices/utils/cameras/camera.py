import abc
from typing import Callable, Optional
import numpy as np


class Camera(abc.ABC):
    """Abstract base class defining a common camera interface."""

    def __init__(self, camera_data: dict, log: Callable[[str], None]) -> None:
        """
        Initialize common parameters and start the camera.

        Args:'
            camera_data: Dict containing at least the keys
                'name', 'fov', 'camera_offset_pos', 'camera_pitch',
                'camera_yaw', 'processing_device', 'frame_rotation'.
            log: Logging function, e.g. `print` or logger.
        """
        self.name: str = camera_data["name"]
        self.fov: np.ndarray = camera_data["fov"]
        self.camera_offset_pos: np.ndarray = camera_data["camera_offset_pos"]
        self.camera_pitch: float = camera_data["camera_pitch"]
        self.camera_yaw: float = camera_data["camera_yaw"]
        self.processing_device: str = camera_data["processing_device"]
        self.frame_rotation: int = camera_data["frame_rotation"]
        self.log = log
        self.cap = None

        self._start_camera()

    @abc.abstractmethod
    def _start_camera(self) -> None:
        """Open or start whatever backend is needed for this camera."""
        pass

    @abc.abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Retrieve a frame (rotated by `frame_rotation`) or None.

        Returns:
            The latest frame, or None on failure/end-of-stream.
        """
        pass

    def get_processing_device(self) -> str:
        """Returns which device (CPU/GPU/TPU) this camera will use."""
        return self.processing_device

    def get_camera_offset_pos(self) -> np.ndarray:
        """Returns the 3D offset position of the camera."""
        return self.camera_offset_pos

    def get_camera_pitch(self) -> float:
        """Returns the pitch (in degrees or radians) of the camera."""
        return self.camera_pitch

    def get_camera_yaw(self) -> float:
        """Returns the yaw (in degrees or radians) of the camera."""
        return self.camera_yaw

    def get_fov(self) -> np.ndarray:
        """Returns the (width, height) resolution this camera is set to."""
        return self.fov

    def get_name(self) -> str:
        """Returns the human‐readable name of this camera."""
        return self.name
