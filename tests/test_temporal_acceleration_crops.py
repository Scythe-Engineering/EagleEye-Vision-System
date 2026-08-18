"""Checks for perspective-aligned temporal acceleration crops."""

from threading import Lock

import cv2
import numpy as np
from pupil_apriltags import Detector

from src.main_operations.definitions.detect_apriltags import (
    DetectApriltagsDefinition,
)
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
    search_region = AprilTagDetector._segment_search_region(crop, full_frame_from_crop)

    np.testing.assert_allclose(restored, source_quad, atol=1e-4)
    np.testing.assert_allclose(search_region, source_quad, atol=1e-4)


def test_perspective_crop_preserves_decodable_tag_winding() -> None:
    """Rectification does not mirror projected AprilTag pixels."""
    tag = cv2.imread(
        "src/webui/assets/apriltags/tag36_11_00001.webp", cv2.IMREAD_GRAYSCALE
    )
    assert tag is not None
    tag = cv2.resize(tag, (160, 160), interpolation=cv2.INTER_NEAREST)
    canonical = np.full((300, 300), 255, dtype=np.uint8)
    canonical[70:230, 70:230] = tag
    canonical_corners = np.array(
        [[0, 0], [299, 0], [299, 299], [0, 299]], dtype=np.float32
    )
    projected_quad = np.array(
        [[100, 300], [300, 300], [300, 100], [100, 100]], dtype=np.float32
    )
    frame_from_canonical = cv2.getPerspectiveTransform(
        canonical_corners, projected_quad[[3, 2, 1, 0]]
    )
    frame = cv2.warpPerspective(
        canonical, frame_from_canonical, (400, 400), borderValue=255
    )

    crop, _ = TemporalAccelerationPreprocessorRustDefinition._perspective_crop(
        frame, projected_quad
    )
    detections = Detector(families="tag36h11", quad_decimate=1).detect(crop)

    assert [detection.tag_id for detection in detections] == [1]


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


def test_detector_fallback_visualizes_only_the_full_frame() -> None:
    """A full-frame fallback replaces stale ROI visualization regions."""
    detector = AprilTagDetector.__new__(AprilTagDetector)
    detector._detect_lock = Lock()
    detector._last_search_regions = []
    detector.run_detection = lambda _images: []
    crop = np.zeros((10, 10), dtype=np.uint8)
    full_frame = np.zeros((20, 30), dtype=np.uint8)

    detector.detect([(crop, np.array([2, 3]))], full_frame)
    regions = detector.get_last_search_regions()

    assert len(regions) == 1
    np.testing.assert_array_equal(regions[0], [[0, 0], [29, 0], [29, 19], [0, 19]])


def test_detector_stays_with_temporal_regions_when_a_tag_is_found() -> None:
    """One temporal detection is enough to skip the full-frame fallback."""
    detector = AprilTagDetector.__new__(AprilTagDetector)
    detector._detect_lock = Lock()
    detector._last_search_regions = []
    calls: list[object] = []
    detector.run_detection = lambda images: calls.append(images) or [object()]
    crop = np.zeros((10, 10), dtype=np.uint8)
    full_frame = np.zeros((20, 30), dtype=np.uint8)

    detector.detect([(crop, np.array([2, 3]))], full_frame)

    assert len(calls) == 1
    assert len(detector.get_last_search_regions()) == 1


def test_detector_uses_full_frame_when_no_temporal_regions_exist() -> None:
    """An empty temporal search falls back to the supplied full frame."""
    detector = AprilTagDetector.__new__(AprilTagDetector)
    detector._detect_lock = Lock()
    detector._last_search_regions = []
    detector.run_detection = lambda _images: []
    full_frame = np.zeros((20, 30), dtype=np.uint8)

    detector.detect([], full_frame)
    regions = detector.get_last_search_regions()

    assert len(regions) == 1
    np.testing.assert_array_equal(regions[0], [[0, 0], [29, 0], [29, 19], [0, 19]])


def test_apriltag_visualization_draws_oriented_search_regions_in_red() -> None:
    """AprilTag visualization outlines temporal search regions in red."""
    search_region = np.array([[50, 20], [80, 50], [50, 80], [20, 50]], dtype=np.float32)

    class DetectorStub:
        def get_last_search_regions(self) -> list[np.ndarray]:
            return [search_region]

    operation = DetectApriltagsDefinition.__new__(DetectApriltagsDefinition)
    operation.detector = DetectorStub()
    operation.last_detections = None
    operation.last_detections_lock = Lock()

    visualization = operation.visualize(np.zeros((100, 100, 3), dtype=np.uint8))

    np.testing.assert_array_equal(visualization[20, 50], [0, 0, 255])
