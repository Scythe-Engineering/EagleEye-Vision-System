"""Synchronous CPU/CUDA YOLO object-detection pipeline operation."""

from __future__ import annotations

from threading import Lock
from typing import Any

import cv2
import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.main_operations.modules.object_detection.yolo_detection.implementation import (
    Detection,
    ObjectDetectionImplementation,
)
from src.utils.device_registry import DeviceRegistry
from src.utils.model_library import ModelLibrary


class ObjectDetectionDefinition(OperationInstance):
    """Run one managed Ultralytics YOLO detection model synchronously."""

    def __init__(
        self,
        model_id: str,
        device_id: str,
        device_registry: DeviceRegistry,
        model_library: ModelLibrary,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        image_size: int = 0,
    ) -> None:
        """Initialize a per-operation detector model.

        Args:
            model_id: Stable ID from the managed model library.
            device_id: Canonical startup registry ID (``cpu`` or ``cuda:N``).
            device_registry: Injected immutable device inventory.
            model_library: Injected managed model library.
            confidence_threshold: Minimum detection confidence.
            iou_threshold: Non-maximum-suppression IoU threshold.
            max_detections: Maximum detections returned per frame.
            image_size: Optional square inference-size override; zero uses model
                or export metadata.
        """
        self.delegate = ObjectDetectionImplementation(
            model_id=model_id,
            device_id=device_id,
            device_registry=device_registry,
            model_library=model_library,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            image_size=image_size,
        )
        self.last_detections: list[Detection] | None = None
        self.last_detections_lock = Lock()
        self.class_colors: dict[int, tuple[int, int, int]] = {}

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Apply per-inference settings without restarting the pipeline.

        Args:
            json_config: Configured parameters, possibly a partial subset.
        """
        self.delegate.update_live_settings(
            confidence_threshold=json_config.get("confidence_threshold"),
            iou_threshold=json_config.get("iou_threshold"),
            max_detections=json_config.get("max_detections"),
        )

    def run(self, frame: np.ndarray) -> list[Detection]:
        """Run detection and retain a snapshot for visualization."""
        detections = self.delegate.run(frame)
        with self.last_detections_lock:
            self.last_detections = detections
        return detections

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Draw the latest normalized detections on a frame."""
        with self.last_detections_lock:
            detections = self.last_detections
        if not detections:
            return frame

        height, width = frame.shape[:2]
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            class_id = detection["class_id"]
            color = self.class_colors.get(class_id)
            if color is None:
                color = self._color_for_class(class_id)
                self.class_colors[class_id] = color
            pixel_box = (
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height),
            )
            cv2.rectangle(
                frame,
                (pixel_box[0], pixel_box[1]),
                (pixel_box[2], pixel_box[3]),
                color,
                3,
            )
            class_label = detection.get("class_name", f"Class {class_id}")
            label = f"{class_label}: {detection['confidence']:.2f}"
            self._draw_label(frame, pixel_box[0], pixel_box[1], label, color)
        return frame

    @staticmethod
    def _color_for_class(class_id: int) -> tuple[int, int, int]:
        """Derive a deterministic bright BGR color for one class ID."""
        hue = (class_id * 47) % 180
        color_pixel = np.array([[[hue, 200, 255]]], dtype=np.uint8)
        blue, green, red = cv2.cvtColor(color_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return int(blue), int(green), int(red)

    @staticmethod
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
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
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
