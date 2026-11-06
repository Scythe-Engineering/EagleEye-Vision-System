"""Behaviour checks for the Rust AprilTag detector."""

from pathlib import Path

import cv2
import numpy as np
from custom_apriltag_detector import CustomAprilTagDetector

FIXTURES = Path(__file__).parent / "fixtures"


def _tag_scene() -> np.ndarray:
    """Render tag 1 with a white quiet zone under a mild perspective warp."""
    tag = cv2.imread(
        "src/webui/assets/apriltags/tag36_11_00001.webp", cv2.IMREAD_GRAYSCALE
    )
    assert tag is not None
    canonical = np.full((300, 300), 255, dtype=np.uint8)
    canonical[70:230, 70:230] = cv2.resize(
        tag, (160, 160), interpolation=cv2.INTER_NEAREST
    )
    source = np.array([[0, 0], [299, 0], [299, 299], [0, 299]], dtype=np.float32)
    target = np.array([[60, 70], [330, 50], [340, 320], [50, 300]], dtype=np.float32)
    warp = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(canonical, warp, (400, 380), borderValue=255)


def _corners(detection) -> np.ndarray:
    return np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)


def test_recovery_still_runs_after_another_tag_decodes() -> None:
    """A rendered frame where the primary pass decodes tag 17 but not 19 and 20."""
    gray = cv2.imread(
        str(FIXTURES / "apriltag_recovery_crop.png"), cv2.IMREAD_GRAYSCALE
    )
    assert gray is not None

    detections = CustomAprilTagDetector(quad_decimate=2.0).run(gray)

    assert sorted(detection.tag_id for detection in detections) == [17, 19, 20]


def test_partially_observed_crop_matches_full_frame_corners() -> None:
    """Masked crops that extend past the frame edge still decode the same corners."""
    frame = _tag_scene()
    detector = CustomAprilTagDetector(quad_decimate=2.0)
    (full,) = detector.run(frame)
    offset = 40
    height, width = frame.shape
    crop = np.full((height + 2 * offset, width + 2 * offset), 255, dtype=np.uint8)
    crop[offset : offset + frame.shape[0], offset : offset + frame.shape[1]] = frame
    crop_to_frame = [1.0, 0.0, -offset, 0.0, 1.0, -offset, 0.0, 0.0, 1.0]

    (masked,) = detector.run(
        crop, source_map=crop_to_frame, source_shape=frame.shape[:2]
    )

    assert masked.tag_id == full.tag_id == 1
    np.testing.assert_allclose(_corners(masked) - offset, _corners(full), atol=0.05)


def test_refine_recovers_corners_from_a_rough_external_quad() -> None:
    """refine() re-decodes and grid-fits corners supplied by another detector."""
    frame = _tag_scene()
    detector = CustomAprilTagDetector(quad_decimate=1.0)
    (reference,) = detector.run(frame)
    rough = _corners(reference) + np.array(
        [[0.6, -0.4], [-0.5, 0.5], [0.4, 0.6], [-0.6, -0.3]]
    )

    refined = detector.refine(frame, rough.ravel().tolist())

    assert refined is not None and refined.tag_id == reference.tag_id
    error = np.linalg.norm(_corners(refined) - _corners(reference), axis=1)
    assert error.max() < 0.1
    assert detector.refine(np.full_like(frame, 255), rough.ravel().tolist()) is None
