"""Shared visualization for normalized object-detection results."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def draw_detections(
    frame: np.ndarray,
    detections: list[dict[str, Any]] | None,
    class_colors: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    """Draw normalized detections on ``frame`` in place and return it."""
    if not detections:
        return frame

    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_id = detection["class_id"]
        if class_id not in class_colors:
            class_colors[class_id] = _color_for_class(class_id)
        color = class_colors[class_id]
        pixel_box = (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )
        cv2.rectangle(frame, pixel_box[:2], pixel_box[2:], color, 3)
        class_label = detection.get("class_name", f"Class {class_id}")
        _draw_label(
            frame,
            pixel_box[0],
            pixel_box[1],
            f"{class_label}: {detection['confidence']:.2f}",
            color,
        )
    return frame


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    """Derive a deterministic bright BGR color for one class ID."""
    hue = (class_id * 47) % 180
    color_pixel = np.array([[[hue, 200, 255]]], dtype=np.uint8)
    blue, green, red = cv2.cvtColor(color_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(blue), int(green), int(red)


def _draw_label(
    frame: np.ndarray,
    x: int,
    y: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a readable label immediately above a bounding box."""
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    text_y = max(y - 8, text_height + baseline + 4)
    cv2.rectangle(
        frame,
        (x, text_y - text_height - baseline - 4),
        (x + text_width + 6, text_y + baseline + 2),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + 3, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
    )
