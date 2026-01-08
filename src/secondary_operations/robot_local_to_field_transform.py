from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from src.secondary_operations.base_class import SecondaryOperation


class RobotLocalToFieldTransform(SecondaryOperation):
    def __init__(self) -> None:
        """Initialize robot-local to field transform operation.

        This operation converts detection positions expressed in the robot's
        local coordinate frame into field coordinates using the latest robot
        pose transform.
        """
        self._latest_robot_transform: np.ndarray | None = None

    def back_propagate_input(self, robot_transform: Any) -> None:
        """Receive the latest robot transform via back propagation.

        Args:
            robot_transform: Robot pose as a 4x4 world-from-robot transform.
        """
        matrix = np.asarray(robot_transform, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("Robot transform must be a finite 4x4 matrix.")
        self._latest_robot_transform = matrix

    @staticmethod
    def _extract_local_position(detection: Dict[str, Any]) -> np.ndarray | None:
        """Extract a valid local position from a detection.

        Args:
            detection: Detection dictionary expected to contain `position_3d`.

        Returns:
            Position as a NumPy array of shape (3,), or None if unavailable/invalid.
        """
        if not isinstance(detection, dict):
            return None
        local = detection.get("position_3d")
        if local is None:
            return None
        try:
            local_array = np.asarray(local, dtype=float)
        except Exception:
            return None
        if local_array.shape != (3,) or not np.all(np.isfinite(local_array)):
            return None
        return local_array

    def _transform_position(self, local_position: np.ndarray) -> np.ndarray:
        """Transform a robot-local position to field coordinates.

        Args:
            local_position: Position in robot-local frame, shape (3,).

        Returns:
            Position in field coordinates, shape (3,).
        """
        assert self._latest_robot_transform is not None
        rotation_matrix = self._latest_robot_transform[:3, :3]
        translation_vector = self._latest_robot_transform[:3, 3]
        return rotation_matrix @ local_position + translation_vector

    def run(
        self, detections: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]] | None:
        """Convert robot-local detection positions to field coordinates.

        Args:
            detections: List of detection dictionaries.

        Returns:
            Updated detections list with `position_3d` in field coordinates.
        """
        if detections is None:
            return None
        if self._latest_robot_transform is None:
            return detections

        transformed_detections: List[Dict[str, Any]] = []
        for detection in detections:
            local_position = self._extract_local_position(detection)
            if local_position is None:
                transformed_detections.append(detection)
                continue

            world_position = self._transform_position(local_position)
            if not np.all(np.isfinite(world_position)):
                transformed_detections.append(detection)
                continue

            updated_detection = detection.copy()
            updated_detection["position_robot"] = local_position.tolist()
            updated_detection["position_3d"] = world_position.tolist()
            transformed_detections.append(updated_detection)

        return transformed_detections
