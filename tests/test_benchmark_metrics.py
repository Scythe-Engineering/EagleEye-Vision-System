"""Analytic tests for benchmark scoring."""

import math

import numpy as np

from benchmarks.metrics import (
    Detection,
    TruthTag,
    availability,
    match_detections,
    pose_errors,
    summary_stats,
)

C = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def test_detection_duplicate_wrong_id_and_empty_denominators() -> None:
    result = match_detections(
        [Detection(1, C), Detection(1, C), Detection(9, C)], [TruthTag(1, C)], 0.1
    )
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)
    assert result["duplicate_detections"] == 1
    assert result["wrong_ids"] == 1
    empty = match_detections([], [], 1)
    assert empty["precision"] is None and empty["eligible_recall"] is None


def test_pose_wrap_stats_and_availability() -> None:
    truth = np.eye(4)
    estimate = np.eye(4)
    angle = math.radians(359)
    estimate[:2, :2] = [
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ]
    error = pose_errors(estimate.tolist(), truth.tolist())
    assert math.isclose(error["yaw_error_rad"], math.radians(-1), abs_tol=1e-9)
    assert summary_stats([])["rmse"] is None
    status = availability([True, False, False, True], [0, 10, 20, 30], [True] * 4)
    assert status["longest_missing_frames"] == 2
