"""Yaw-constrained PnP: a known robot heading fixes rotation, position is solved exactly."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.main_operations.definitions.pnp_camera_localization import (
    _NWU_FROM_EDN,
    PnpCameraLocalizationDefinition,
)
from src.main_operations.modules.apriltags.pnp_localization import PnpLocalization

CAMERA_MATRIX = np.array(
    [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
TAG_HALF_SIZE = 0.0825
CAMERA_POSITION = np.array([1.0, 2.0, 0.8])
ROBOT_YAW = np.deg2rad(30.0)


def _rotation_for_yaw(yaw: float) -> np.ndarray:
    """R_field_from_camera for a flat robot with identity camera extrinsics."""
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    field_from_robot = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
    )
    return field_from_robot @ _NWU_FROM_EDN


def _scene() -> tuple[PnpLocalization, list[SimpleNamespace]]:
    """Two vertical tags 3m ahead of a camera at a known field pose and yaw."""
    rotation = _rotation_for_yaw(ROBOT_YAW)
    forward = rotation[:, 2]
    lateral = np.array([-forward[1], forward[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])

    corners_by_id: dict[int, np.ndarray] = {}
    for tag_id, side in ((1, -0.5), (2, 0.5)):
        center = CAMERA_POSITION + 3.0 * forward + side * lateral
        corners_by_id[tag_id] = np.array(
            [
                center - TAG_HALF_SIZE * lateral + TAG_HALF_SIZE * up,
                center + TAG_HALF_SIZE * lateral + TAG_HALF_SIZE * up,
                center + TAG_HALF_SIZE * lateral - TAG_HALF_SIZE * up,
                center - TAG_HALF_SIZE * lateral - TAG_HALF_SIZE * up,
            ]
        ).astype(np.float32)

    camera_from_field = rotation.T

    def project(points: np.ndarray) -> np.ndarray:
        camera_points = (points - CAMERA_POSITION) @ camera_from_field.T
        return np.column_stack(
            (
                600.0 * camera_points[:, 0] / camera_points[:, 2] + 320.0,
                600.0 * camera_points[:, 1] / camera_points[:, 2] + 240.0,
            )
        ).astype(np.float32)

    estimator = PnpLocalization(
        camera_matrix=CAMERA_MATRIX,
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        apriltag_map={
            tag_id: SimpleNamespace(global_corners=points)
            for tag_id, points in corners_by_id.items()
        },
    )
    detections = [
        SimpleNamespace(tag_id=tag_id, corners=project(points))
        for tag_id, points in corners_by_id.items()
    ]
    return estimator, detections


def _definition() -> PnpCameraLocalizationDefinition:
    definition = object.__new__(PnpCameraLocalizationDefinition)
    definition.camera_bus_id = "0"
    definition.camera_config_registry = None
    definition.pose_estimator, definition._detections = _scene()
    return definition


def test_constrained_solve_recovers_position_and_imposes_rotation() -> None:
    definition = _definition()

    output = definition.run(
        {"detections": definition._detections, "robot_yaw": float(ROBOT_YAW)}
    )

    pose = output["camera_pose"]
    assert pose is not None
    np.testing.assert_allclose(pose[:3, 3], CAMERA_POSITION, atol=1e-3)
    np.testing.assert_allclose(pose[:3, :3], _rotation_for_yaw(ROBOT_YAW), atol=1e-4)

    tag_count, mean_distance, reprojection_error = output["pose_meta"]
    assert tag_count == 2.0
    assert abs(mean_distance - float(np.hypot(3.0, 0.5))) < 0.01
    assert reprojection_error < 0.1


def test_constrained_and_unconstrained_agree_on_clean_data() -> None:
    definition = _definition()

    constrained = definition.run(
        {"detections": definition._detections, "robot_yaw": float(ROBOT_YAW)}
    )
    unconstrained = definition.run({"detections": definition._detections})

    np.testing.assert_allclose(
        constrained["camera_pose"], unconstrained["camera_pose"], atol=5e-3
    )


def test_wrong_yaw_still_imposes_the_supplied_rotation() -> None:
    """The gyro is trusted over the tags: the output rotation is exactly the input yaw."""
    definition = _definition()
    wrong_yaw = ROBOT_YAW + 0.1

    output = definition.run(
        {"detections": definition._detections, "robot_yaw": float(wrong_yaw)}
    )

    assert output["camera_pose"] is not None
    np.testing.assert_allclose(
        output["camera_pose"][:3, :3], _rotation_for_yaw(wrong_yaw), atol=1e-4
    )


def test_non_finite_yaw_falls_back_to_unconstrained() -> None:
    definition = _definition()

    output = definition.run(
        {"detections": definition._detections, "robot_yaw": float("nan")}
    )
    unconstrained = definition.run({"detections": definition._detections})

    np.testing.assert_allclose(
        output["camera_pose"], unconstrained["camera_pose"], atol=1e-6
    )
