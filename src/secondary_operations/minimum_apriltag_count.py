from __future__ import annotations

from collections.abc import Sized
from typing import Any

from src.config.utils.operation import SKIP_PIPELINE_CYCLE
from src.main_operations.definitions.base.base_class import OperationInstance


class MinimumApriltagCount(OperationInstance):
    """Require a minimum number of AprilTag detections before continuing a frame."""

    def __init__(self, minimum_detections: int = 2) -> None:
        """Initialize the gate with the required detection count.

        Args:
            minimum_detections: Number of detections required to continue processing.
        """
        self.minimum_detections = self._validate_minimum(minimum_detections)

    def run(self, detections: Sized | None) -> Any:
        """Pass detections through or skip the rest of the current pipeline cycle.

        Args:
            detections: AprilTag detections produced by the detector.

        Returns:
            The unchanged detections when enough tags were found, otherwise the
            pipeline-control sentinel that advances processing to the next frame.
        """
        if detections is None or len(detections) < self.minimum_detections:
            return SKIP_PIPELINE_CYCLE
        return detections

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Apply a live update to the minimum detection count."""
        if "minimum_detections" in json_config:
            self.minimum_detections = self._validate_minimum(
                json_config["minimum_detections"]
            )

    @staticmethod
    def _validate_minimum(value: Any) -> int:
        """Return a valid positive integer threshold."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("minimum_detections must be an integer of at least 1")
        return value
