"""PnP quality metrics feed the robot-side pose estimator's standard deviations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.main_operations.definitions.pnp_camera_localization import (
    PnpCameraLocalizationDefinition,
)
from src.main_operations.modules.apriltags.pnp_localization import PnpLocalization

CAMERA_MATRIX = np.array(
    [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
TAG_HALF_SIZE = 0.0825


def _tag_corners(center: tuple[float, float, float]) -> np.ndarray:
    """Four coplanar corners centred on *center*, facing the camera."""
    x, y, z = center
    return np.array(
        [
            [x - TAG_HALF_SIZE, y + TAG_HALF_SIZE, z],
            [x + TAG_HALF_SIZE, y + TAG_HALF_SIZE, z],
            [x + TAG_HALF_SIZE, y - TAG_HALF_SIZE, z],
            [x - TAG_HALF_SIZE, y - TAG_HALF_SIZE, z],
        ],
        dtype=np.float32,
    )


def _project(points: np.ndarray) -> np.ndarray:
    """Pinhole projection with no distortion, so reprojection error must be ~0."""
    return np.column_stack(
        (
            600.0 * points[:, 0] / points[:, 2] + 320.0,
            600.0 * points[:, 1] / points[:, 2] + 240.0,
        )
    ).astype(np.float32)


def _estimator_and_detections() -> tuple[PnpLocalization, list[SimpleNamespace]]:
    """Two tags placed in front of a camera sitting at the field origin."""
    centers = {1: (0.0, 0.0, 3.0), 2: (0.5, 0.0, 4.0)}
    corners = {tag_id: _tag_corners(center) for tag_id, center in centers.items()}
    estimator = PnpLocalization(
        camera_matrix=CAMERA_MATRIX,
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        apriltag_map={
            tag_id: SimpleNamespace(global_corners=points)
            for tag_id, points in corners.items()
        },
    )
    detections = [
        SimpleNamespace(tag_id=tag_id, corners=_project(points))
        for tag_id, points in corners.items()
    ]
    return estimator, detections


def test_pose_meta_reports_tag_count_distance_and_reprojection_error() -> None:
    """A solved pose must include the metrics used by robot-side filtering."""
    estimator, detections = _estimator_and_detections()

    solution = estimator.estimate_pose_from_detections(detections)

    assert solution is not None
    pose, meta = solution
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-3)

    tag_count, mean_distance, reprojection_error = meta
    assert tag_count == 2.0
    expected_distance = (3.0 + float(np.hypot(0.5, 4.0))) / 2.0
    assert abs(mean_distance - expected_distance) < 1e-3
    assert reprojection_error < 0.1


def test_unmapped_tags_produce_no_solution() -> None:
    """Detections absent from the configured field map cannot produce a pose."""
    estimator, _ = _estimator_and_detections()

    assert (
        estimator.estimate_pose_from_detections(
            [SimpleNamespace(tag_id=99, corners=np.zeros((4, 2), dtype=np.float32))]
        )
        is None
    )


def test_failed_solve_still_fills_both_output_ports() -> None:
    """Downstream operations rely on None, not a missing port, when PnP fails."""
    definition = object.__new__(PnpCameraLocalizationDefinition)
    definition.pose_estimator = SimpleNamespace(
        estimate_pose_from_detections=lambda _detections: None
    )

    assert definition.run([]) == {"camera_pose": None, "pose_meta": None}


def test_successful_solve_routes_pose_and_meta_to_separate_ports() -> None:
    """Successful solves must route pose and quality metadata independently."""
    estimator, detections = _estimator_and_detections()
    definition = object.__new__(PnpCameraLocalizationDefinition)
    definition.pose_estimator = estimator

    output = definition.run(detections)

    assert set(output) == {"camera_pose", "pose_meta"}
    np.testing.assert_allclose(output["camera_pose"], np.eye(4), atol=1e-3)
    assert output["pose_meta"][0] == 2.0
