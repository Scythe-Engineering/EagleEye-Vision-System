from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from src.main_operations.definitions.base.base_class import OperationInstance


class RobotLocalToFieldTransform(OperationInstance):
    def __init__(self) -> None:
        """Initialize robot-local to field transform operation.

        This operation converts detection positions expressed in the robot's
        local coordinate frame into field coordinates using the robot
        pose transform.
        """
        self._latest_robot_transform: np.ndarray | None = None

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

    def run(self, input_data: Any) -> List[Dict[str, Any]] | None:
        """Convert robot-local detection positions to field coordinates.

        Args:
            input_data: Input data - dict with 'detections' and optionally 'robot_pose' keys.

        Returns:
            Updated detections list with `position_3d` in field coordinates.
        """
        if isinstance(input_data, dict):
            detections = input_data.get("detections")
            robot_pose = input_data.get("robot_pose")
        else:
            detections = input_data
            robot_pose = None

        if robot_pose is not None:
            matrix = np.asarray(robot_pose, dtype=float)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError("Robot transform must be a finite 4x4 matrix.")
            self._latest_robot_transform = matrix

        if detections is None:
            return None

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
