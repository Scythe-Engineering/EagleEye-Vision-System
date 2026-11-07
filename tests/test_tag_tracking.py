"""Behaviour of AprilTagDetector tag tracking with the Rust backend."""

from __future__ import annotations

import cv2
import numpy as np

from src.main_operations.modules.apriltags.apriltag_detector import AprilTagDetector
from tests.test_custom_apriltag_detector import _tag_scene


def _shifted(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        frame, warp, frame.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=255
    )


def test_tracked_tag_skips_its_segment_and_matches_the_search() -> None:
    """A moving tag is re-found from its prediction; a lost tag is searched again."""
    scene = _tag_scene()
    tracking = AprilTagDetector(rust_backend=True, track_tags=True)
    searching = AprilTagDetector(rust_backend=True)
    searched_segments: list[int] = []
    run_detection = tracking.run_detection

    def spy(images, **kwargs):
        if isinstance(images, list):  # Full-frame fallbacks pass an array.
            searched_segments.append(len(images))
        return run_detection(images, **kwargs)

    tracking.run_detection = spy
    blank = np.full_like(scene, 255)
    frames = [_shifted(scene, 1.5 * i, -0.8 * i) for i in range(4)] + [blank, scene]
    for frame in frames:
        segments = [(frame, np.array([0, 0]))]
        tracked = tracking.detect(segments, frame)
        reference = searching.detect(segments, frame)
        if frame is blank:
            assert not tracked
            continue
        ((tag,), (expected,)) = (tracked, reference)
        assert tag.tag_id == expected.tag_id == 1
        np.testing.assert_allclose(tag.corners, expected.corners, atol=0.1)

    # Frame 0 is searched; frames 1-3 are tracked; the blank frame and the frame
    # after it are searched because the miss dropped the track.
    assert searched_segments == [1, 0, 0, 0, 1, 1]
