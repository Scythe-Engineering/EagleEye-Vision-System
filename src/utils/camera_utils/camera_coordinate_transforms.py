"""Coordinate transforms for camera-local detection points."""

from __future__ import annotations

import numpy as np

from src.utils.camera_utils.camera_config_manager import CameraExtrinsics
from src.utils.quaternion_utils import euler_to_rotation_matrix


# OpenCV camera coordinates are X right, Y down, Z forward. Robot coordinates
# are X forward, Y left, Z up.
_CAMERA_TO_ROBOT_BASIS = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


def build_robot_from_camera_transform(extrinsics: CameraExtrinsics) -> np.ndarray:
    """Build a transform from OpenCV camera coordinates to robot coordinates.

    Args:
        extrinsics: Camera mounting pose in robot coordinates. Positive pitch
            points the camera downward, matching ground-plane configuration.

    Returns:
        A 4x4 transform mapping camera-local points into the robot frame.
    """
    mounting_rotation = euler_to_rotation_matrix(
        pitch=float(extrinsics.pitch),
        yaw=float(extrinsics.yaw),
        roll=float(extrinsics.roll),
    )[:3, :3]

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = mounting_rotation @ _CAMERA_TO_ROBOT_BASIS
    transform[:3, 3] = np.array(
        [extrinsics.x_offset, extrinsics.y_offset, extrinsics.z_offset],
        dtype=float,
    )
    return transform
