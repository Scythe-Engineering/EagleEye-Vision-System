"""Synchronous CPU/CUDA YOLO object-detection pipeline operation."""

from __future__ import annotations

from threading import Lock
from time import monotonic
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
from src.utils.model_library import ModelLibrary, ModelLibraryError


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
            model_id: Stable managed model ID, or empty to wait for an upload.
            device_id: Canonical startup registry ID (``cpu`` or ``cuda:N``).
            device_registry: Injected immutable device inventory.
            model_library: Injected managed model library.
            confidence_threshold: Minimum detection confidence.
            iou_threshold: Non-maximum-suppression IoU threshold.
            max_detections: Maximum detections returned per frame.
            image_size: Optional square inference-size override; zero uses model
                or export metadata.
        """
        self.model_id = model_id
        self.device_id = device_id
        self.device_registry = device_registry
        self.model_library = model_library
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.image_size = image_size
        self.delegate: ObjectDetectionImplementation | None = None
        self._next_model_check = 0.0
        self._empty_slot_ignored_ids = (
            self._compatible_model_ids() if not model_id else set()
        )
        self.last_detections: list[Detection] | None = None
        self.last_detections_lock = Lock()
        self.class_colors: dict[int, tuple[int, int, int]] = {}
        self._load_available_model()

    def _compatible_model_ids(self) -> set[str]:
        """Return model IDs that currently resolve for this detector device."""
        compatible: set[str] = set()
        for model in self.model_library.list_models():
            try:
                self.model_library.resolve_artifact(model.model_id, self.device_id)
            except ModelLibraryError:
                continue
            compatible.add(model.model_id)
        return compatible

    def _load_available_model(self) -> ObjectDetectionImplementation | None:
        """Load the configured or newly compatible model when one is available."""
        if self.delegate is not None:
            return self.delegate
        now = monotonic()
        if now < self._next_model_check:
            return None
        self._next_model_check = now + 1.0

        candidate_ids: tuple[str, ...]
        if self.model_id:
            candidate_ids = (self.model_id,)
        else:
            candidate_ids = tuple(
                model.model_id
                for model in self.model_library.list_models()
                if model.model_id not in self._empty_slot_ignored_ids
            )
        for candidate_id in candidate_ids:
            try:
                self.model_library.resolve_artifact(candidate_id, self.device_id)
            except ModelLibraryError:
                if self.model_id:
                    raise
                continue
            try:
                self.delegate = ObjectDetectionImplementation(
                    model_id=candidate_id,
                    device_id=self.device_id,
                    device_registry=self.device_registry,
                    model_library=self.model_library,
                    confidence_threshold=self.confidence_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections,
                    image_size=self.image_size,
                )
            except (RuntimeError, ValueError):
                if self.model_id:
                    raise
                continue
            self.model_id = candidate_id
            return self.delegate
        return None

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Apply per-inference settings without restarting the pipeline.

        Args:
            json_config: Configured parameters, possibly a partial subset.
        """
        if self.delegate is not None:
            self.delegate.update_live_settings(
                confidence_threshold=json_config.get("confidence_threshold"),
                iou_threshold=json_config.get("iou_threshold"),
                max_detections=json_config.get("max_detections"),
            )
        if "confidence_threshold" in json_config:
            self.confidence_threshold = json_config["confidence_threshold"]
        if "iou_threshold" in json_config:
            self.iou_threshold = json_config["iou_threshold"]
        if "max_detections" in json_config:
            self.max_detections = json_config["max_detections"]

    def run(self, frame: np.ndarray) -> list[Detection]:
        """Run detection, or idle until a compatible model is uploaded.

        Args:
            frame: Camera image to infer on.
        """
        delegate = self._load_available_model()
        detections = [] if delegate is None else delegate.run(frame)
        with self.last_detections_lock:
            self.last_detections = detections
        return detections

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Draw the latest normalized detections on a frame."""
        with self.last_detections_lock:
            detections = self.last_detections
        return draw_detections(frame, detections, self.class_colors)
