"""Tests for NetworkTables WPILib pose publishing helpers."""

from __future__ import annotations

import math

import numpy as np

from src.secondary_operations.publish_to_networktables import _matrix_to_pose3d


def test_matrix_to_pose3d_converts_opencv_camera_axes_to_wpilib_body_axes() -> None:
    """A level robot pose should not publish with a 90 degree roll."""

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    transform[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=float)

    pose = _matrix_to_pose3d(transform)

    assert pose.X() == 1.0
    assert pose.Y() == 2.0
    assert pose.Z() == 3.0
    assert math.isclose(math.degrees(pose.rotation().X()), 0.0, abs_tol=1e-9)
    assert math.isclose(math.degrees(pose.rotation().Y()), 0.0, abs_tol=1e-9)
    assert math.isclose(math.degrees(pose.rotation().Z()), 0.0, abs_tol=1e-9)


def test_matrix_to_pose3d_preserves_wpilib_yaw_after_basis_conversion() -> None:
    """OpenCV camera-axis pose should publish robot yaw in WPILib NWU axes."""

    yaw = math.radians(45.0)
    field_from_robot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    camera_from_robot_axes = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = field_from_robot @ camera_from_robot_axes

    pose = _matrix_to_pose3d(transform)

    assert math.isclose(math.degrees(pose.rotation().X()), 0.0, abs_tol=1e-9)
    assert math.isclose(math.degrees(pose.rotation().Y()), 0.0, abs_tol=1e-9)
    assert math.isclose(math.degrees(pose.rotation().Z()), 45.0, abs_tol=1e-9)
