import json
from typing import Tuple

import numpy as np


def load_camera_parameters(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from a JSON file.

    Args:
        json_path (str): Path to the camera intrinsics JSON file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Camera matrix and distortion coefficients.
    """
    with open(json_path, "r") as file:
        data = json.load(file)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    distortion_coefficients = np.array(
        data["distortion_coefficients"], dtype=np.float64
    )
    return camera_matrix, distortion_coefficients 