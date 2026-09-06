"""Tests for camera-pose to robot-pose conversion using camera extrinsics."""

from __future__ import annotations

import numpy as np
import pytest

from src.secondary_operations.camera_to_robot_pose import CameraToRobotPose
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.quaternion_utils import euler_to_rotation_matrix


def _make_pose(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    """Build a simple world-from-camera pose for assertions."""

    pose = euler_to_rotation_matrix(pitch=0.0, yaw=yaw_deg, roll=0.0)
    pose[:3, 3] = np.array([x, y, z], dtype=float)
    return pose


def test_camera_to_robot_pose_applies_extrinsics(tmp_path) -> None:
    """Camera pose should be converted into robot pose using configured extrinsics."""

    registry = CameraConfigRegistry(base_path=str(tmp_path))
    camera_config = registry.get_config("cam0")
    camera_config.update_extrinsics_live(
        {
            "pitch": 0.0,
            "yaw": 90.0,
            "roll": 0.0,
            "x_offset": 0.5,
            "y_offset": 0.0,
            "z_offset": 0.25,
        }
    )

    operation = CameraToRobotPose(
        camera_bus_id="cam0",
        camera_config_registry=registry,
    )
    camera_pose = _make_pose(x=5.0, y=2.0, z=1.5, yaw_deg=45.0)

    basis = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    robot_from_camera = np.eye(4)
    robot_from_camera[:3, :3] = (
        basis.T @ euler_to_rotation_matrix(0, 90, 0)[:3, :3] @ basis
    )
    robot_from_camera[:3, 3] = basis.T @ np.array([0.5, 0.0, 0.25])
    expected_robot_pose = camera_pose @ CameraToRobotPose._fast_se3_inverse(
        robot_from_camera
    )

    result = operation.run(camera_pose)

    assert result is not None
    np.testing.assert_allclose(result, expected_robot_pose)


def test_camera_to_robot_pose_invalidates_cached_extrinsics_on_update(tmp_path) -> None:
    """Live config updates should rebuild the cached extrinsics transform."""

    registry = CameraConfigRegistry(base_path=str(tmp_path))
    camera_config = registry.get_config("cam0")
    camera_config.update_extrinsics_live(
        {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "x_offset": 1.0,
            "y_offset": 0.0,
            "z_offset": 0.0,
        }
    )

    operation = CameraToRobotPose(
        camera_bus_id="cam0",
        camera_config_registry=registry,
    )
    camera_pose = np.eye(4, dtype=float)

    initial_result = operation.run(camera_pose)

    camera_config.update_extrinsics_live(
        {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "x_offset": 2.0,
            "y_offset": 0.0,
            "z_offset": 0.0,
        }
    )
    operation.update_config({"x_offset": 2.0})
    updated_result = operation.run(camera_pose)

    assert initial_result is not None
    assert updated_result is not None
    assert initial_result[2, 3] == pytest.approx(-1.0)
    assert updated_result[2, 3] == pytest.approx(-2.0)


def test_camera_to_robot_pose_rejects_invalid_input() -> None:
    """Invalid pose inputs should be ignored instead of raising."""

    operation = CameraToRobotPose(camera_bus_id="cam0")

    assert operation.run(None) is None
    assert operation.run(np.array([[1.0, 2.0], [3.0, 4.0]])) is None
