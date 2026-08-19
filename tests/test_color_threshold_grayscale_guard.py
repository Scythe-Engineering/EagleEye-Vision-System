from __future__ import annotations

import numpy as np
import pytest

from src.main_operations.modules.object_detection.color_threshold_detection.implementation import (
    ColorThresholdDetectionImplementation,
)

RED_RANGE = [
    {"name": "red", "class_id": 0, "lower_hsv": [0, 100, 100], "upper_hsv": [10, 255, 255]}
]


def test_grayscale_frame_is_rejected_with_an_actionable_message() -> None:
    """A mono camera must fail loudly here rather than match nothing.

    HSV thresholding a single-channel frame would either raise deep inside
    OpenCV or, worse, silently return zero detections that read as "no target
    visible" instead of "this camera cannot do colour".
    """
    implementation = ColorThresholdDetectionImplementation(color_ranges=RED_RANGE)
    grayscale_frame = np.zeros((64, 64), dtype=np.uint8)

    with pytest.raises(ValueError, match="monochrome"):
        implementation.run(grayscale_frame)


def test_colour_frame_is_accepted() -> None:
    implementation = ColorThresholdDetectionImplementation(color_ranges=RED_RANGE)
    colour_frame = np.zeros((64, 64, 3), dtype=np.uint8)

    detections, mask = implementation.run(colour_frame)

    assert isinstance(detections, list)
    assert mask is not None
