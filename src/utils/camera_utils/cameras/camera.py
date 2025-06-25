import abc
import json
from typing import Callable, Optional

import numpy as np


def load_camera_parameters(
    json_path: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load camera matrix and distortion coefficients from a JSON file.

    Args:
        json_path (str): Path to the camera intrinsics JSON file.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[int, int]]: Camera matrix, distortion coefficients, and image size (width, height).
    """
    with open(json_path, "r") as file:
        data = json.load(file)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    distortion_coefficients = np.array(
        data["distortion_coefficients"], dtype=np.float64
    )

    # Extract image dimensions from calibration data
    if "img_size" in data:
        image_width, image_height = data["img_size"]
    elif "image_width" in data and "image_height" in data:
        image_width, image_height = data["image_width"], data["image_height"]
    else:
        raise ValueError("Image dimensions not found in camera calibration file")

    return camera_matrix, distortion_coefficients, (image_width, image_height)


def calculate_fov_from_camera_matrix(
    camera_matrix: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Calculate field of view angles from camera matrix and image dimensions.

    Args:
        camera_matrix (np.ndarray): 3x3 camera intrinsic matrix.
        image_size (tuple[int, int]): Image dimensions as (width, height).

    Returns:
        np.ndarray: FOV angles in degrees as [horizontal_fov, vertical_fov].
    """
    image_width, image_height = image_size

    # Extract focal lengths from camera matrix
    focal_length_x = camera_matrix[0, 0]  # fx
    focal_length_y = camera_matrix[1, 1]  # fy

    # Calculate FOV angles in radians then convert to degrees
    horizontal_fov_radians = 2 * np.arctan(image_width / (2 * focal_length_x))
    vertical_fov_radians = 2 * np.arctan(image_height / (2 * focal_length_y))

    # Convert to degrees for compatibility with existing code
    horizontal_fov_degrees = np.degrees(horizontal_fov_radians)
    vertical_fov_degrees = np.degrees(vertical_fov_radians)

    return np.array([horizontal_fov_degrees, vertical_fov_degrees])


class Camera(abc.ABC):
    """Abstract base class defining a common camera interface."""

    def __init__(
        self, camera_data: dict, log: Callable[[str], None], camera_intrinsics_path: str
    ) -> None:
        """
        Initialize common parameters and start the camera.

        Args:
            camera_data: Dict containing at least the keys
                'name', 'camera_offset_pos', 'camera_pitch',
                'camera_yaw', 'processing_device', 'frame_rotation'.
            log: Logging function, e.g. `print` or logger.
            camera_intrinsics_path: Path to the camera intrinsics JSON file.
        """
        self.name: str = camera_data["name"]
        self.camera_offset_pos: np.ndarray = camera_data["camera_offset_pos"]
        self.camera_pitch: float = camera_data["camera_pitch"]
        self.camera_yaw: float = camera_data["camera_yaw"]
        self.frame_rotation: int = camera_data["frame_rotation"]

        self.camera_matrix, self.distortion_coefficients, self.image_size = (
            load_camera_parameters(camera_intrinsics_path)
        )

        # Calculate FOV angles from camera matrix instead of using stored data
        self.fov: np.ndarray = calculate_fov_from_camera_matrix(
            self.camera_matrix, self.image_size
        )

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
        """Returns the field of view angles in degrees as [horizontal_fov, vertical_fov]."""
        return self.fov

    def get_name(self) -> str:
        """Returns the human‐readable name of this camera."""
        return self.name
