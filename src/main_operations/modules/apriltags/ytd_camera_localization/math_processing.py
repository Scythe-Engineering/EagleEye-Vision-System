import cv2
import numpy as np
from typing import Iterable, Tuple


def estimate_tag_distance_and_horizontal_angle(
    corners_px: Iterable[Iterable[float]],
    tag_local_corners: Iterable[Iterable[float]],
    distortion_coefficients: Iterable[float],
    camera_matrix: np.ndarray,
    camera_pitch: float,
) -> Tuple[float, float]:
    """Compute distance (meters) from camera to the tag center using homography.

    This routine is optimized for speed and avoids expensive rotation recovery.

    Args:
        corners_px: Iterable of four (u, v) image pixel coordinates in order [TL, TR, BR, BL].
        tag_local_corners: Iterable of four (x, y, z[, 1]) object points ordered [TL, TR, BR, BL].
        distortion_coefficients: Iterable of distortion coefficients.
        camera_matrix: 3x3 camera intrinsic matrix.
        camera_pitch: Pitch of the camera in radians.

    Returns:
        Tuple containing:
        - Distance from camera to the tag center in the same units as `tag_local_corners`.
        - Horizontal angle to the tag center in radians.
    """

    object_points = np.asarray(tag_local_corners, dtype=np.float32)
    image_points = np.asarray(corners_px, dtype=np.float32)

    success, _, translation_vector = cv2.solvePnP(
        object_points, image_points, camera_matrix, distortion_coefficients
    )

    if not success:
        raise ValueError("Failed to estimate tag distance and horizontal angle.")

    tx, tz = (
        float(translation_vector[0]),
        float(translation_vector[2]),
    )

    distance = float(np.sqrt(tx**2 + tz**2))

    try:
        distance = np.cos(camera_pitch) * distance
    except TypeError:
        raise TypeError(
            f"Failed to estimate tag distance and horizontal angle. camera_pitch: {camera_pitch}, distance: {distance}"
        )

    horizontal_angle = float(np.arctan2(tx, tz))

    return distance, horizontal_angle
