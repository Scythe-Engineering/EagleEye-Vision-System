"""Cached tag-set geometry must never change a solve, and the cache must stay bounded."""

from __future__ import annotations

import numpy as np

from src.main_operations.modules.apriltags import pnp_localization
from tests.test_pnp_pose_meta import _estimator_and_detections


def test_cached_geometry_matches_a_fresh_estimator_in_either_tag_order() -> None:
    """Warm solves, including a reversed detection order, equal cold ones exactly."""
    warm, detections = _estimator_and_detections()
    reversed_detections = detections[::-1]
    for _ in range(2):
        warm.estimate_pose_from_detections(detections)
        warm.estimate_pose_from_detections(reversed_detections)

    for frame in (detections, reversed_detections):
        cold, _ = _estimator_and_detections()
        warm_pose, warm_meta = warm.estimate_pose_from_detections(frame)
        cold_pose, cold_meta = cold.estimate_pose_from_detections(frame)
        np.testing.assert_array_equal(warm_pose, cold_pose)
        assert warm_meta == cold_meta


def test_tag_set_cache_evicts_least_recently_used_sets(monkeypatch) -> None:
    """Unusual tag-id combinations cannot grow the cache past its bound."""
    monkeypatch.setattr(pnp_localization, "TAG_SET_CACHE_SIZE", 2)
    estimator, detections = _estimator_and_detections()

    for frame in ([detections[0]], [detections[1]], detections, [detections[1]]):
        estimator.estimate_pose_from_detections(frame)

    assert list(estimator._tag_set_cache) == [(1, 2), (2,)]
