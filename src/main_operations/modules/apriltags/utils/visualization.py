"""Shared AprilTag overlay for live previews and saved benchmark diagnostics."""

from collections.abc import Iterable

import cv2
import numpy as np


def draw_apriltag_overlay(
    frame: np.ndarray,
    detections: Iterable[tuple[int, np.ndarray]],
    search_regions: Iterable[np.ndarray],
) -> np.ndarray:
    """Draw actual searched regions red and successful detections green with IDs."""
    image = frame.copy()
    for region in search_regions:
        cv2.polylines(image, [np.rint(region).astype(np.int32)], True, (0, 0, 255), 2)
    for tag_id, points in detections:
        corners = np.asarray(points).astype(np.int32)
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        center = (int(corners[:, 0].mean()), int(corners[:, 1].mean()))
        cv2.putText(
            image,
            f"ID: {tag_id}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
    return image
