"""Focused checks for opt-in full-frame AprilTag threading."""

from __future__ import annotations

import cv2
import numpy as np
from pupil_apriltags import Detector

from src.main_operations.definitions.detect_apriltags import DetectApriltagsDefinition
from src.main_operations.modules.apriltags import apriltag_detector
from src.main_operations.modules.apriltags.apriltag_detector import AprilTagDetector


def test_full_frame_threads_do_not_change_roi_threads_or_fallback(monkeypatch) -> None:
    """ROIs keep nthreads while direct and fallback full frames use the override."""
    calls: list[int] = []

    class RecordingDetector:
        def __init__(self, *, nthreads: int, **_kwargs: object) -> None:
            self.nthreads = nthreads

        def detect(self, _image: np.ndarray) -> list[object]:
            calls.append(self.nthreads)
            return []

    monkeypatch.setattr(apriltag_detector, "Detector", RecordingDetector)
    detector = AprilTagDetector(
        nthreads=1,
        full_frame_nthreads=2,
        large_roi_decimate=3,
    )
    roi = np.zeros((32, 32), dtype=np.uint8)
    large_roi = np.zeros((128, 128), dtype=np.uint8)
    full_frame = np.zeros((256, 256), dtype=np.uint8)

    detector.run_detection([(roi, np.zeros(2)), (large_roi, np.zeros(2))])
    detector.run_detection(full_frame)
    detector.detect([(roi, np.zeros(2))], full_frame)

    assert calls == [1, 1, 2, 1, 2]


def test_zero_full_frame_threads_reuses_base_detector_after_live_updates(
    monkeypatch,
) -> None:
    """Zero removes the override, including through the operation config wrapper."""

    class RecordingDetector:
        def __init__(self, *, nthreads: int, **_kwargs: object) -> None:
            self.nthreads = nthreads

        def detect(self, _image: np.ndarray) -> list[object]:
            return []

    monkeypatch.setattr(apriltag_detector, "Detector", RecordingDetector)
    detector = AprilTagDetector(nthreads=1, full_frame_nthreads=2, large_roi_decimate=0)
    detector.update_parameters(full_frame_nthreads=0)

    assert detector.full_frame_nthreads == 0
    assert detector._full_frame_detector is None
    detector.run_detection(np.zeros((32, 32), dtype=np.uint8))
    assert detector.detector.nthreads == 1

    operation = DetectApriltagsDefinition(
        nthreads=1, full_frame_nthreads=2, large_roi_decimate=0
    )
    operation.update_config({"full_frame_nthreads": 0})
    assert operation.detector.full_frame_nthreads == 0
    assert operation.detector._full_frame_detector is None


def test_cropped_tag_corners_match_direct_detection_and_full_metadata() -> None:
    """Segment corner mapping matches direct detection without changing metadata."""
    tag = cv2.imread(
        "src/webui/assets/apriltags/tag36_11_00001.webp", cv2.IMREAD_GRAYSCALE
    )
    assert tag is not None
    tag = cv2.resize(tag, (160, 160), interpolation=cv2.INTER_NEAREST)
    frame = np.full((300, 300), 255, dtype=np.uint8)
    frame[70:230, 70:230] = tag

    raw_detection = Detector(families="tag36h11", quad_decimate=1).detect(frame)[0]
    detector = AprilTagDetector(quad_decimate=1, large_roi_decimate=0)
    direct = detector.run_detection(frame)
    cropped = detector.run_detection([(frame[40:260, 40:260], np.array([40, 40]))])

    assert direct is not None and cropped is not None
    assert len(direct) == len(cropped) == 1
    np.testing.assert_allclose(cropped[0].corners, direct[0].corners, atol=0.2)
    np.testing.assert_allclose(direct[0].center, raw_detection.center - 0.5, atol=1e-6)
    shift = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(direct[0].homography, shift @ raw_detection.homography)
