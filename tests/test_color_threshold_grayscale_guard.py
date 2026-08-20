from __future__ import annotations

import numpy as np
import pytest

from src.main_operations.modules.object_detection.color_threshold_detection.implementation import (
    ColorThresholdDetectionImplementation,
)

RED_RANGE = [
    {
        "name": "red",
        "class_id": 0,
        "lower_hsv": [0, 100, 100],
        "upper_hsv": [10, 255, 255],
    }
]


@pytest.mark.parametrize("shape", [(64, 64), (64, 64, 1), (64, 64, 4)])
def test_non_bgr_frame_is_rejected_with_an_actionable_message(
    shape: tuple[int, ...],
) -> None:
    """Reject frame shapes that OpenCV cannot treat as three-channel BGR."""
    implementation = ColorThresholdDetectionImplementation(color_ranges=RED_RANGE)
    invalid_frame = np.zeros(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="three-channel BGR"):
        implementation.run(invalid_frame)


def test_colour_frame_is_accepted() -> None:
    """Accept valid three-channel BGR input."""
    implementation = ColorThresholdDetectionImplementation(color_ranges=RED_RANGE)
    colour_frame = np.zeros((64, 64, 3), dtype=np.uint8)

    detections, mask = implementation.run(colour_frame)

    assert isinstance(detections, list)
    assert mask.shape == (320, 320, 3)
    assert mask.dtype == np.uint8
