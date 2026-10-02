"""Deterministic scoring primitives for synthetic benchmark results."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A normalized full-image tag detection."""

    tag_id: int
    corners: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TruthTag:
    """A rendered tag observation and its scoring category."""

    tag_id: int
    corners: tuple[tuple[float, float], ...]
    eligible: bool = True
    category: str = "eligible"


def _corner_error(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    """Return mean corresponding-corner pixel distance."""
    if len(left) != 4 or len(right) != 4:
        raise ValueError("tag corners must contain exactly four points")
    return float(
        np.mean(
            np.linalg.norm(
                np.asarray(left, dtype=float) - np.asarray(right, dtype=float), axis=1
            )
        )
    )


def match_detections(
    detections: Sequence[Detection], truth: Sequence[TruthTag], tolerance_px: float
) -> dict[str, Any]:
    """Match detections one-to-one by identity and corner proximity."""
    if tolerance_px < 0 or not math.isfinite(tolerance_px):
        raise ValueError("tolerance_px must be finite and nonnegative")
    candidates = sorted(
        (_corner_error(det.corners, tag.corners), di, ti)
        for di, det in enumerate(detections)
        for ti, tag in enumerate(truth)
        if det.tag_id == tag.tag_id
    )
    used_d: set[int] = set()
    used_t: set[int] = set()
    matches: list[dict[str, Any]] = []
    for error, di, ti in candidates:
        if error <= tolerance_px and di not in used_d and ti not in used_t:
            used_d.add(di)
            used_t.add(ti)
            matches.append(
                {"detection_index": di, "truth_index": ti, "corner_error_px": error}
            )
    unmatched_d = [i for i in range(len(detections)) if i not in used_d]
    unmatched_t = [i for i in range(len(truth)) if i not in used_t]
    duplicate_indices = [
        i
        for i in unmatched_d
        if any(detections[i].tag_id == truth[j].tag_id for j in used_t)
    ]
    wrong_id_indices = [
        i
        for i in unmatched_d
        if i not in duplicate_indices
        and any(
            _corner_error(detections[i].corners, truth[j].corners) <= tolerance_px
            and detections[i].tag_id != truth[j].tag_id
            for j in range(len(truth))
        )
    ]
    eligible = {i for i, tag in enumerate(truth) if tag.eligible}
    tp = sum(item["truth_index"] in eligible for item in matches)
    fn = len(eligible - used_t)
    fp = len(unmatched_d)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "eligible_recall": recall,
        "matches": matches,
        "corner_errors_px": [item["corner_error_px"] for item in matches],
        "duplicate_detections": len(duplicate_indices),
        "duplicate_indices": duplicate_indices,
        "wrong_ids": len(wrong_id_indices),
        "wrong_id_indices": wrong_id_indices,
        "unmatched_detection_indices": unmatched_d,
        "unmatched_truth_indices": unmatched_t,
    }


def pose_errors(
    estimate: Sequence[Sequence[float]], truth: Sequence[Sequence[float]]
) -> dict[str, float]:
    """Calculate translation, relative rotation, and wrapped yaw errors."""
    estimated = np.asarray(estimate, dtype=float)
    expected = np.asarray(truth, dtype=float)
    if estimated.shape != (4, 4) or expected.shape != (4, 4):
        raise ValueError("poses must be 4x4")
    if not np.isfinite(estimated).all() or not np.isfinite(expected).all():
        raise ValueError("poses must be finite")
    delta = estimated[:3, 3] - expected[:3, 3]
    relative = expected[:3, :3].T @ estimated[:3, :3]
    rotation = math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    estimate_yaw = math.atan2(estimated[1, 0], estimated[0, 0])
    truth_yaw = math.atan2(expected[1, 0], expected[0, 0])
    yaw = (estimate_yaw - truth_yaw + math.pi) % (2.0 * math.pi) - math.pi
    return {
        "translation_xy_m": float(np.linalg.norm(delta[:2])),
        "translation_3d_m": float(np.linalg.norm(delta)),
        "rotation_rad": rotation,
        "yaw_error_rad": yaw,
        "yaw_absolute_error_rad": abs(yaw),
    }


def summary_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    """Return count, RMSE, median, p95, p99, and maximum, null when empty."""
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {
            "count": 0,
            "rmse": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    if not np.isfinite(data).all():
        raise ValueError("summary values must be finite")
    return {
        "count": int(data.size),
        "rmse": float(np.sqrt(np.mean(data * data))),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def availability(
    valid: Sequence[bool],
    timestamps_ns: Sequence[int] | None = None,
    eligible: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Summarize availability, contiguous missing intervals, and recoveries."""
    count = len(valid)
    if timestamps_ns is not None and len(timestamps_ns) != count:
        raise ValueError("timestamps and validity lengths differ")
    if eligible is not None and len(eligible) != count:
        raise ValueError("eligibility and validity lengths differ")
    intervals: list[dict[str, int]] = []
    start: int | None = None
    for index, value in enumerate([*valid, True]):
        if index < count and not value and start is None:
            start = index
        elif value and start is not None:
            intervals.append(
                {"start_frame": start, "end_frame": index - 1, "frames": index - start}
            )
            start = None
    recoveries: list[dict[str, int | None]] = []
    if eligible is not None:
        for index in range(count):
            if eligible[index] and (index == 0 or not eligible[index - 1]):
                recovered = next((j for j in range(index, count) if valid[j]), None)
                delay = (
                    None
                    if recovered is None
                    else (
                        (timestamps_ns[recovered] - timestamps_ns[index])
                        if timestamps_ns
                        else recovered - index
                    )
                )
                recoveries.append(
                    {
                        "visibility_frame": index,
                        "recovered_frame": recovered,
                        "delay_ns" if timestamps_ns else "delay_frames": delay,
                    }
                )
    eligible_total = sum(eligible) if eligible is not None else count
    eligible_valid = sum(
        ok and (eligible[i] if eligible is not None else True)
        for i, ok in enumerate(valid)
    )
    return {
        "frames": count,
        "valid_frames": sum(valid),
        "availability": sum(valid) / count if count else None,
        "eligible_frames": eligible_total,
        "eligible_availability": eligible_valid / eligible_total
        if eligible_total
        else None,
        "missing_intervals": intervals,
        "longest_missing_frames": max(
            (item["frames"] for item in intervals), default=0
        ),
        "recoveries": recoveries,
    }
