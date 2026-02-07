from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.device_management_utils.compute_pool import ComputePool
from src.webui.web_server import EagleEyeInterface


class RobotLocalToFieldTransform(OperationInstance):
    """Convert robot-local detections to field coordinates.

    Inputs:
        Detections in robot-local coordinates and optional robot pose transform.
    Outputs:
        Detections in field coordinates with updated position data.
    """

    def __init__(
        self, web_interface: EagleEyeInterface, compute_pool: ComputePool
    ) -> None:
        """Initialize robot-local to field transform operation.

        Args:
            web_interface: Web interface for runtime updates.
            compute_pool: Compute pool available for device operations.
        """
        self.web_interface = web_interface
        self.compute_pool = compute_pool
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
        """Transform detections from robot-local to field coordinates.

        Args:
            input_data: Dict with `detections` list and optional `robot_pose` 4x4 matrix.

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
