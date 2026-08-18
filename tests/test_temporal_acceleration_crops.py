"""Checks for perspective-aligned temporal acceleration crops."""

from threading import Lock

import numpy as np

from src.main_operations.definitions.temporal_acceleration_preprocessor_rust import (
    TemporalAccelerationPreprocessorRustDefinition,
)
from src.main_operations.modules.apriltags.apriltag_detector import AprilTagDetector


def test_perspective_crop_maps_detection_corners_back_to_full_frame() -> None:
    """Rectified crop coordinates recover their projected full-frame positions."""
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    source_quad = np.array([[10, 20], [70, 10], [80, 80], [20, 90]], dtype=np.float32)

    crop, full_frame_from_crop = (
        TemporalAccelerationPreprocessorRustDefinition._perspective_crop(
            frame, source_quad.flatten()
        )
    )
    side = crop.shape[0]
    crop_corners = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype=np.float32,
    )

    restored = AprilTagDetector._map_segment_corners(crop_corners, full_frame_from_crop)
    np.testing.assert_allclose(restored, source_quad, atol=1e-4)


def test_visualization_uses_projected_quad_instead_of_axis_aligned_bounds() -> None:
    """Only the rotated predicted region remains at full brightness."""
    operation = TemporalAccelerationPreprocessorRustDefinition.__new__(
        TemporalAccelerationPreprocessorRustDefinition
    )
    operation._last_visualization_quads = [
        np.array([[50, 20], [80, 50], [50, 80], [20, 50]], dtype=np.float32)
    ]
    operation._last_visualization_quads_lock = Lock()
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)

    visualization = operation.visualize(frame)

    np.testing.assert_array_equal(visualization[50, 50], [100, 100, 100])
    np.testing.assert_array_equal(visualization[25, 25], [30, 30, 30])
