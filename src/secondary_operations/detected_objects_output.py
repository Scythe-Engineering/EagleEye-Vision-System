from collections.abc import Iterable
from typing import Any, Dict, List, Sequence

import numpy as np

from src.webui.web_server import EagleEyeInterface
from src.secondary_operations.base_class import SecondaryOperation


class DetectedObjectsOutput(SecondaryOperation):
    def __init__(self, web_interface: EagleEyeInterface) -> None:
        """Initialize detected objects output operation.

        Args:
            web_interface: Web interface responsible for sending detections.
        """
        self.web_interface = web_interface
        self._last_signature: tuple | None = None

    def _is_valid_position(self, position: Any) -> bool:
        return (
            isinstance(position, Iterable)
            and len(position) == 3
            and all(
                isinstance(coord, (int, float)) and np.isfinite(coord)
                for coord in position
            )
        )

    def _build_detection_payload(
        self, detection: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        position = detection.get("position_3d")
        if not self._is_valid_position(position):
            return None
        payload: Dict[str, Any] = {
            "position_3d": [float(coord) for coord in position],  # type: ignore[arg-type]
        }
        if "class_id" in detection:
            payload["class_id"] = detection["class_id"]
        if "class_name" in detection:
            payload["class_name"] = detection["class_name"]
        if "confidence" in detection and isinstance(
            detection["confidence"], (int, float)
        ):
            confidence_value = float(detection["confidence"])
            if np.isfinite(confidence_value):
                payload["confidence"] = confidence_value
        return payload

    def _compute_signature(self, detections: Sequence[Dict[str, Any]]) -> tuple:
        signature_items = []
        for detection in detections:
            position = detection.get("position_3d")
            if not self._is_valid_position(position):
                continue
            class_identifier = detection.get("class_name", detection.get("class_id"))
            confidence_value = detection.get("confidence")
            signature_items.append(
                (
                    tuple(float(coord) for coord in position),  # type: ignore[arg-type]
                    class_identifier,
                    None if confidence_value is None else float(confidence_value),
                )
            )
        return tuple(signature_items)

    def run(
        self, detections: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]] | None:
        """Send detections to web interface for 3D visualization.

        Args:
            detections: List of detection dictionaries.

        Returns:
            The detections list to allow pipeline chaining.
        """
        if detections is None:
            return None
        payloads = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            payload = self._build_detection_payload(detection)
            if payload is not None:
                payloads.append(payload)
        signature = self._compute_signature(payloads)
        if self._last_signature == signature:
            return detections
        self.web_interface.update_detected_objects(payloads)
        self._last_signature = signature
        return detections
