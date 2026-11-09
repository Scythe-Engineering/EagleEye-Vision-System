"""Re-find tags seen in the previous frame from their predicted corners."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .lazy_segment import segment_shape_and_mapping

if TYPE_CHECKING:
    from .custom_native import CustomNativeDetector


@dataclass
class _Track:
    """Last two full-frame corner sets of one tag, OpenCV pixel-center coordinates."""

    corners: np.ndarray
    previous: np.ndarray | None


class TagTracker:
    """Predict each tracked tag with constant velocity and confirm it natively.

    Only tags found in the previous frame are tracked. A miss drops the track, so the
    tag returns to the normal temporal-ROI search on the same frame.
    """

    def __init__(self) -> None:
        """Start with no tracked tags."""
        self._tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        """Forget all tracks, e.g. after a detector reconfiguration."""
        self._tracks.clear()

    def find(
        self, native: CustomNativeDetector, frame: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Return confirmed full-frame corners by tag ID for this frame's tracks.

        Args:
            native: CustomNativeDetector used to confirm each prediction.
            frame: Full BGR or grayscale frame.

        Returns:
            Confirmed shape-(4, 2) corners by tag ID; missed tags are omitted.
        """
        if not self._tracks:
            return {}
        tag_ids = list(self._tracks)
        predicted = [self._predict(track) for track in self._tracks.values()]
        confirmed = native.track_many(frame, predicted, tag_ids)
        return {
            tag_id: corners
            for tag_id, corners in zip(tag_ids, confirmed)
            if corners is not None
        }

    def update(self, detections: Iterable[Any]) -> None:
        """Track exactly the tags in this frame's final detections.

        Args:
            detections: Detections with ``tag_id`` and full-frame ``corners``.
        """
        tracks = {}
        for detection in detections:
            corners = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
            old = self._tracks.get(detection.tag_id)
            tracks[detection.tag_id] = _Track(
                corners.copy(), None if old is None else old.corners
            )
        self._tracks = tracks

    @staticmethod
    def _predict(track: _Track) -> np.ndarray:
        """Constant-velocity corners for this frame, or the last corners when new."""
        if track.previous is None:
            return track.corners
        return 2.0 * track.corners - track.previous


def segment_center(shape: tuple[int, ...], mapping: np.ndarray) -> tuple[float, float]:
    """Full-frame position of a segment's center for a 2D offset or 3x3 map."""
    height, width = shape[:2]
    x, y = (width - 1) * 0.5, (height - 1) * 0.5
    # Scalar math: a few floats per segment are far cheaper than numpy calls.
    values = np.asarray(mapping, dtype=np.float64).tolist()
    if len(values) == 2:
        return x + values[0], y + values[1]
    (a, b, c), (d, e, f), (g, h, i) = values
    z = g * x + h * y + i
    return (a * x + b * y + c) / z, (d * x + e * y + f) / z


def uncovered_segments(
    segments: list,
    tracked: dict[int, np.ndarray],
    is_searchable: Callable[[tuple[int, ...]], bool],
) -> list:
    """Drop the segments centred on a tracked tag, without building them.

    Args:
        segments: Temporal segments, eager or lazy.
        tracked: Confirmed full-frame corners by tag ID.
        is_searchable: Whether a segment of this shape would be searched at all;
            other segments are kept so the search skips them as before.

    Returns:
        The segments still to search.
    """
    # Segments are centred on one projected tag each; half an edge tolerates pose
    # lag. A few tags and segments per frame: plain floats beat numpy calls here.
    circles = []
    for corners in tracked.values():
        points = corners.tolist()
        (x0, y0), _, (x2, y2), _ = points
        center_x = sum(point[0] for point in points) / 4.0
        center_y = sum(point[1] for point in points) / 4.0
        dx, dy = x0 - x2, y0 - y2
        radius = 0.5 * math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)
        circles.append((center_x, center_y, radius))

    def covered(segment) -> bool:
        shape, mapping = segment_shape_and_mapping(segment)
        if not is_searchable(shape):
            return False
        try:
            x, y = segment_center(shape, mapping)
        except ZeroDivisionError:
            # Center maps to a projective pole: keep the segment searched.
            return False
        return any(
            math.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) < radius
            for cx, cy, radius in circles
        )

    return [segment for segment in segments if not covered(segment)]
