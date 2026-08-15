"""Tests for camera-local detection transformation."""

from __future__ import annotations

import pytest

from src.secondary_operations.camera_local_to_robot_transform import (
    CameraLocalToRobotTransform,
)
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry


def test_camera_local_detection_transforms_to_robot_coordinates(tmp_path) -> None:
    """Camera basis, mounting rotation, and translation should all be applied."""
    registry = CameraConfigRegistry(base_path=str(tmp_path))
    registry.get_config("cam0").update_extrinsics_live(
        {
            "pitch": 45.0,
            "yaw": 0.0,
            "roll": 0.0,
            "x_offset": 0.5,
            "y_offset": 0.25,
            "z_offset": 2.0,
        }
    )
    operation = CameraLocalToRobotTransform("cam0", registry)

    result = operation.run([{"position_3d": [0.0, 0.0, 2.0**1.5]}])

    assert result[0]["position_camera"] == pytest.approx([0.0, 0.0, 2.0**1.5])
    assert result[0]["position_3d"] == pytest.approx([2.5, 0.25, 0.0])


def test_non_finite_extrinsics_are_rejected(tmp_path) -> None:
    """Invalid calibration must fail loudly instead of emitting NaN positions."""
    registry = CameraConfigRegistry(base_path=str(tmp_path))
    registry.get_config("cam0").update_extrinsics_live(
        {
            "pitch": float("nan"),
            "yaw": 0.0,
            "roll": 0.0,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "z_offset": 0.0,
        }
    )
    operation = CameraLocalToRobotTransform("cam0", registry)

    with pytest.raises(ValueError, match="non-finite mounting extrinsics"):
        operation.run([{"position_3d": [0.0, 0.0, 1.0]}])
