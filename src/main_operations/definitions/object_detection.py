"""Synchronous CPU/CUDA YOLO object-detection pipeline operation."""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.main_operations.modules.object_detection.utils.detection_visualization import (
    draw_detections,
)
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
        return draw_detections(frame, detections, self.class_colors)
