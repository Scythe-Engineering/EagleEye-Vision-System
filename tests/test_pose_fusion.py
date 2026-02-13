import numpy as np
import pytest
from unittest.mock import MagicMock
from src.secondary_operations.pose_fusion import PoseFusion


def create_test_pose(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    """Create a 4x4 transformation matrix with given translation and yaw rotation.

    Args:
        x: X translation.
        y: Y translation.
        z: Z translation.
        yaw: Yaw rotation in radians.

    Returns:
        4x4 transformation matrix.
    """
    pose = np.eye(4)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    pose[0, 0] = cos_yaw
    pose[0, 1] = -sin_yaw
    pose[1, 0] = sin_yaw
    pose[1, 1] = cos_yaw

    return pose


def test_single_pose():
    """Test fusion with single pose input."""
    fusion = PoseFusion(
        web_interface=MagicMock(),
        compute_pool=MagicMock()
    )
    pose = create_test_pose(1.0, 2.0, 0.0, 0.5)

    result = fusion.run(pose)

    assert result is not None
    np.testing.assert_array_almost_equal(result, pose)


def test_two_similar_poses():
    """Test fusion with two similar poses."""
    fusion = PoseFusion(
        web_interface=MagicMock(),
        compute_pool=MagicMock()
    )
    pose1 = create_test_pose(1.0, 2.0, 0.0, 0.5)
    pose2 = create_test_pose(1.1, 2.1, 0.0, 0.52)

    result = fusion.run({"pose_0": pose1, "pose_1": pose2})

    assert result is not None
    assert result.shape == (4, 4)


def test_multiple_poses_with_outlier():
    """Test fusion with 5 poses where 1 is an outlier."""
    fusion = PoseFusion(
        web_interface=MagicMock(),
        compute_pool=MagicMock(),
        outlier_threshold=0.5
    )

    pose1 = create_test_pose(1.0, 2.0, 0.0, 0.5)
    pose2 = create_test_pose(1.05, 2.05, 0.0, 0.51)
    pose3 = create_test_pose(0.95, 1.95, 0.0, 0.49)
    pose4 = create_test_pose(1.02, 2.02, 0.0, 0.505)
    outlier = create_test_pose(5.0, 5.0, 0.0, 1.5)

    result = fusion.run({
        "pose_0": pose1,
        "pose_1": pose2,
        "pose_2": pose3,
        "pose_3": pose4,
        "pose_4": outlier,
    })

    assert result is not None

    result_x = result[0, 3]
    result_y = result[1, 3]

    assert 0.9 < result_x < 1.1
    assert 1.9 < result_y < 2.1


def test_none_input():
    """Test fusion with None input."""
    fusion = PoseFusion(
        web_interface=MagicMock(),
        compute_pool=MagicMock()
    )

    result = fusion.run(None)

    assert result is None


def test_invalid_pose_shape():
    """Test fusion rejects invalid pose shapes."""
    fusion = PoseFusion(
        web_interface=MagicMock(),
        compute_pool=MagicMock()
    )

    result = fusion.run(np.array([[1, 2], [3, 4]]))

    assert result is None


def test_mixed_valid_and_invalid():
    """Test fusion filters out invalid poses."""
    fusion = PoseFusion(
        web_interface=MagicMock(),
        compute_pool=MagicMock()
    )

    pose1 = create_test_pose(1.0, 2.0, 0.0, 0.5)
    invalid = np.array([[1, 2], [3, 4]])

    result = fusion.run({"pose_0": pose1, "pose_1": invalid})

    assert result is not None
    np.testing.assert_array_almost_equal(result, pose1)


def test_weighted_averaging():
    """Test that fusion properly weights poses by distance from cluster."""
    fusion = PoseFusion(
        web_interface=MagicMock(),
        compute_pool=MagicMock()
    )

    center_pose = create_test_pose(1.0, 2.0, 0.0, 0.5)
    nearby_pose = create_test_pose(1.05, 2.05, 0.0, 0.51)

    result = fusion.run({"pose_0": center_pose, "pose_1": nearby_pose})

    assert result is not None

    result_x = result[0, 3]
    result_y = result[1, 3]

    assert 1.0 <= result_x <= 1.05
    assert 2.0 <= result_y <= 2.05


def test_update_config():
    """Test update_config method."""
    fusion = PoseFusion(
        web_interface=MagicMock(),
        compute_pool=MagicMock(),
        outlier_threshold=1.0,
        rotation_weight=0.5
    )

    fusion.update_config({"outlier_threshold": 2.0, "rotation_weight": 0.8})

    assert fusion.outlier_threshold == 2.0
    assert fusion.rotation_weight == 0.8
