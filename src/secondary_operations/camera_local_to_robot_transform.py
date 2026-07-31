"""Transform camera-local detections into robot-local coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.camera_coordinate_transforms import (
    build_robot_from_camera_transform,
)


class CameraLocalToRobotTransform(OperationInstance):
    """Convert detection positions from the camera frame to the robot frame."""

    def __init__(
        self,
        camera_bus_id: str,
        camera_config_registry: CameraConfigRegistry | None = None,
    ) -> None:
        """Initialize the transform using the selected camera's extrinsics.

        Args:
            camera_bus_id: Camera bus ID used to resolve mounting extrinsics.
            camera_config_registry: Injected shared camera config registry.
        """
        self.camera_bus_id = str(camera_bus_id)
        self.camera_config_registry = camera_config_registry

    def update_config(self, json_config: dict[str, object]) -> None:
        """Update runtime-configurable operation parameters.

        Args:
            json_config: Configuration dictionary with updated values.
        """
        if "camera_bus_id" in json_config:
            self.camera_bus_id = str(json_config["camera_bus_id"])

    def run(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform valid ``position_3d`` values into robot coordinates.

        Args:
            detections: Detection dictionaries containing camera-local positions.

        Returns:
            Copies of valid detections with camera and robot positions retained.
        """
        if self.camera_config_registry is None:
            raise ValueError(
                "Camera config registry is required for camera-to-robot transform."
            )

        extrinsics = self.camera_config_registry.get_config(
            self.camera_bus_id
        ).extrinsics
        transform = build_robot_from_camera_transform(extrinsics)

        output: list[dict[str, Any]] = []
        for detection in detections:
            position = detection.get("position_3d")
            try:
                camera_position = np.asarray(position, dtype=float)
            except (TypeError, ValueError):
                output.append(detection)
                continue

            if camera_position.shape != (3,) or not np.all(
                np.isfinite(camera_position)
            ):
                output.append(detection)
                continue

            robot_position = transform[:3, :3] @ camera_position + transform[:3, 3]
            updated_detection = detection.copy()
            updated_detection["position_camera"] = camera_position.tolist()
            updated_detection["position_3d"] = robot_position.tolist()
            output.append(updated_detection)

        return output
