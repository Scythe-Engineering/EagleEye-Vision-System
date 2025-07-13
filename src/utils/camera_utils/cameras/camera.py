import abc
import json
import os
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
        self, camera_name: str, camera_calibration_folder: str | None, log: Callable[[str], None] = print
    ) -> None:
        """
        Initialize common parameters and start the camera.

        Args:
            camera_name: Name of the camera.
            camera_calibration_folder: Path to the camera calibration folder. If None, the camera will not be calibrated.
            log: Logging function, e.g. `print` or logger. Defaults to `print`.
        """
        self.name: str = camera_name
        self.is_calibrated: bool = False

        if camera_calibration_folder is not None:
            self.is_calibrated = True
            
            self.camera_matrix, self.distortion_coefficients, self.image_size = (
                load_camera_parameters(os.path.join(camera_calibration_folder, "intrinsics.json"))
            )

            self.fov: np.ndarray = calculate_fov_from_camera_matrix(
                self.camera_matrix, self.image_size
            )
        else:
            log(f"Camera: {self.name} created without intrinsics calibration")
            
        try:
            with open(os.path.join(str(camera_calibration_folder), "extrinsics.json"), "r") as file:
                extrinsics = json.load(file)
                
            self.camera_offset_pos: np.ndarray = extrinsics["camera_offset_pos"]
            self.camera_pitch: float = extrinsics["camera_pitch"]
            self.camera_yaw: float = extrinsics["camera_yaw"]
            self.frame_rotation: int = extrinsics["frame_rotation"]
        except FileNotFoundError:
            self.log(f"Camera: {self.name} created without extrinsics calibration")
            
        self.camera_ready: bool = False

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
    
    def calibrate_camera(self, camera_calibration_folder: str) -> None:
        """Calibrate the camera."""
        self.camera_matrix, self.distortion_coefficients, self.image_size = (
            load_camera_parameters(os.path.join(camera_calibration_folder, "intrinsics.json"))
        )
        self.fov = calculate_fov_from_camera_matrix(
            self.camera_matrix, self.image_size
        )
        self.is_calibrated = True
        self.log(f"Camera: {self.name} calibrated")

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
