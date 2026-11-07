"""Re-find tags seen in the previous frame from their predicted corners."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
        self._tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        """Forget all tracks, e.g. after a detector reconfiguration."""
        self._tracks.clear()

    def find(self, native, frame: np.ndarray) -> dict[int, np.ndarray]:
        """Return confirmed full-frame corners by tag ID for this frame's tracks.

        Args:
            native: CustomNativeDetector used to confirm each prediction.
            frame: Full BGR or grayscale frame.
        """
        found = {}
        height, width = frame.shape[:2]
        for tag_id, track in self._tracks.items():
            predicted = self._predict(track)
            # The window covers the white quiet zone and the edge-snap retry.
            edge = float(np.linalg.norm(predicted[0] - predicted[2])) / np.sqrt(2.0)
            margin = edge / 8.0 * 1.5 + 8.0
            x0 = max(0, int(np.floor(predicted[:, 0].min() - margin)))
            y0 = max(0, int(np.floor(predicted[:, 1].min() - margin)))
            x1 = min(width, int(np.ceil(predicted[:, 0].max() + margin)) + 1)
            y1 = min(height, int(np.ceil(predicted[:, 1].max() + margin)) + 1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            window = self._gray(frame[y0:y1, x0:x1])
            offset = np.array([x0, y0], dtype=np.float64)
            # Native corners use pixel-edge coordinates.
            prior = (predicted - offset + 0.5).ravel().tolist()
            confirmed = native.track(window, prior, tag_id)
            if confirmed is not None:
                found[tag_id] = confirmed - 0.5 + offset
        return found

    def update(self, detections) -> None:
        """Track exactly the tags in this frame's final detections."""
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

    @staticmethod
    def _gray(window: np.ndarray) -> np.ndarray:
        # Windows change size every frame, so let OpenCV allocate the small output.
        return window if window.ndim == 2 else cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)


def segment_center(image: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Full-frame position of a segment's center for a 2D offset or 3x3 map."""
    height, width = image.shape[:2]
    center = np.array([(width - 1) * 0.5, (height - 1) * 0.5])
    mapping = np.asarray(mapping, dtype=np.float64)
    if mapping.shape == (2,):
        return center + mapping
    x, y, z = mapping @ np.array([center[0], center[1], 1.0])
    return np.array([x / z, y / z])
