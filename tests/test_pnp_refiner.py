"""The native LM refiner must match OpenCV's closely and degrade to it when unusable."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.main_operations.modules.apriltags import pnp_localization
from tests.test_pnp_pose_meta import CAMERA_MATRIX, _estimator_and_detections

pnp_refiner = pytest.importorskip("pnp_refiner")

DISTORTION = np.array([-0.06, 0.012, 0.0005, -0.0005, 0.0])
OBJECT_POINTS = np.array(
    [[-0.08, 0.08, 0.0], [0.08, 0.08, 0.0], [0.08, -0.08, 0.0], [-0.08, -0.08, 0.0]]
)


def test_native_refinement_matches_opencv_from_a_perturbed_start() -> None:
    """Both refiners reach the same pose from the same rough initializer."""
    true_rvec = np.array([[0.2], [-0.3], [0.1]])
    true_tvec = np.array([[0.1], [-0.05], [2.5]])
    image_points = cv2.projectPoints(
        OBJECT_POINTS, true_rvec, true_tvec, CAMERA_MATRIX, DISTORTION
    )[0].reshape(-1, 2)
    image_points += np.random.default_rng(7).normal(0, 0.3, image_points.shape)
    start_rvec = true_rvec + 0.03
    start_tvec = true_tvec + 0.05

    expected = cv2.solvePnPRefineLM(
        OBJECT_POINTS,
        image_points,
        CAMERA_MATRIX,
        DISTORTION,
        start_rvec.copy(),
        start_tvec.copy(),
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1e-9),
    )
    rvec, tvec = start_rvec.copy(), start_tvec.copy()
    refiner = pnp_refiner.PnpRefiner(CAMERA_MATRIX, DISTORTION)
    assert refiner.refine(OBJECT_POINTS, image_points, rvec, tvec, 10, 1e-9)

    np.testing.assert_allclose(rvec, expected[0], atol=1e-6)
    np.testing.assert_allclose(tvec, expected[1], atol=1e-6)


def test_unsupported_distortion_models_keep_opencv_refinement() -> None:
    """Thin-prism or tilted models fall back to OpenCV instead of failing."""
    estimator, detections = _estimator_and_detections()
    estimator.distortion_coefficients = np.zeros((14, 1))
    assert estimator._create_rust_refiner() is None

    estimator._rust_refiner = None
    assert estimator.estimate_pose_from_detections(detections) is not None


def test_missing_native_module_keeps_opencv_refinement(monkeypatch) -> None:
    """Without the built module the estimator solves exactly as before."""
    monkeypatch.setattr(pnp_localization, "RustPnpRefiner", None)
    estimator, detections = _estimator_and_detections()

    assert estimator._rust_refiner is None
    pose, meta = estimator.estimate_pose_from_detections(detections)
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-3)
    assert meta[0] == 2.0
