"""Tests for camera pose publishing to the WebUI."""

from __future__ import annotations

import numpy as np

from src.secondary_operations.camera_pose_output import CameraPoseOutput
from tests.utils.dummy_dependencies import FakeEagleEyeInterface


def test_camera_pose_output_publishes_valid_pose_and_passthrough() -> None:
    """Valid camera pose should be published once and passed through."""

    web_interface = FakeEagleEyeInterface()
    operation = CameraPoseOutput(camera_bus_id="cam0", web_interface=web_interface)
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=float)

    result = operation.run(pose)

    assert result is pose
    assert len(web_interface.camera_poses) == 1
    camera_bus_id, published_pose = web_interface.camera_poses[0]
    assert camera_bus_id == "cam0"
    np.testing.assert_allclose(published_pose, pose)


def test_camera_pose_output_deduplicates_unchanged_pose() -> None:
    """Identical consecutive poses should not republish."""

    web_interface = FakeEagleEyeInterface()
    operation = CameraPoseOutput(camera_bus_id="cam0", web_interface=web_interface)
    pose = np.eye(4, dtype=float)

    first_result = operation.run(pose)
    second_result = operation.run(pose.copy())

    assert first_result is pose
    assert second_result is not None
    assert len(web_interface.camera_poses) == 1


def test_camera_pose_output_rejects_invalid_pose() -> None:
    """Invalid camera poses should be ignored instead of raising."""

    web_interface = FakeEagleEyeInterface()
    operation = CameraPoseOutput(camera_bus_id="cam0", web_interface=web_interface)

    assert operation.run(None) is None
    assert operation.run(np.array([[1.0, 2.0], [3.0, 4.0]])) is None

    invalid_pose = np.eye(4, dtype=float)
    invalid_pose[0, 0] = np.nan
    assert operation.run(invalid_pose) is None
    assert web_interface.camera_poses == []
