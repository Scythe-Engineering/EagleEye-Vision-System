"""Checks for perspective-aligned temporal acceleration crops."""

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
