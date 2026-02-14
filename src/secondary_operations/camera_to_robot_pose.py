from __future__ import annotations

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.quaternion_utils import euler_to_rotation_matrix


class CameraToRobotPose(OperationInstance):
    """Convert camera pose to robot pose using camera extrinsics.

    This operation consumes a camera pose matrix and converts it to robot pose
    by applying the inverse of the camera extrinsic transform.
    """

    def __init__(
        self,
        camera_bus_id: str,
        camera_config_registry: CameraConfigRegistry | None = None,
    ) -> None:
        """Initialize the camera-to-robot pose transform operation.

        Args:
            camera_bus_id: Camera bus ID used to resolve extrinsics.
            camera_config_registry: Injected shared camera config registry.
        """
        self.camera_bus_id = str(camera_bus_id)
        self.camera_config_registry = camera_config_registry

    @staticmethod
    def _fast_se3_inverse(transform: np.ndarray) -> np.ndarray:
        """Compute fast analytical inverse for a 4x4 SE(3) transform.

        Args:
            transform: 4x4 SE(3) transformation matrix.

        Returns:
            4x4 inverse transformation matrix.
        """
        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        rotation_inverse = rotation.T
        translation_inverse = -rotation_inverse @ translation

        inverse_transform = np.empty((4, 4), dtype=float)
        inverse_transform.fill(0.0)
        inverse_transform[3, 3] = 1.0
        inverse_transform[:3, :3] = rotation_inverse
        inverse_transform[:3, 3] = translation_inverse
        return inverse_transform

    def _build_camera_to_robot_transform(self) -> np.ndarray:
        """Build ``T_robot_from_camera`` from current camera extrinsics.

        Returns:
            4x4 transform mapping points from camera frame to robot frame.

        Raises:
            ValueError: If camera config registry is unavailable.
        """
        if self.camera_config_registry is None:
            raise ValueError("camera_config_registry is required for CameraToRobotPose")

        camera_config = self.camera_config_registry.get_config(self.camera_bus_id)
        extrinsics = camera_config.extrinsics

        transform = euler_to_rotation_matrix(
            pitch=float(extrinsics.pitch),
            yaw=float(extrinsics.yaw),
            roll=float(extrinsics.roll),
        )
        transform[:3, 3] = np.array(
            [
                float(extrinsics.x_offset),
                float(extrinsics.y_offset),
                float(extrinsics.z_offset),
            ],
            dtype=float,
        )
        return transform

    def _build_inverse_transform(self) -> np.ndarray:
        """Build ``T_camera_from_robot`` by inverting camera extrinsics.

        Returns:
            4x4 transform mapping robot frame to camera frame.
        """
        robot_from_camera = self._build_camera_to_robot_transform()
        return self._fast_se3_inverse(robot_from_camera)

    def update_config(self, json_config: dict[str, object]) -> None:
        """Update runtime-configurable operation parameters.

        Args:
            json_config: Configuration dictionary with updated values.
        """
        if "camera_bus_id" in json_config:
            self.camera_bus_id = str(json_config["camera_bus_id"])

    def run(self, camera_pose: np.ndarray | None) -> np.ndarray | None:
        """Convert camera pose to robot pose.

        Args:
            camera_pose: 4x4 camera pose transform, or None.

        Returns:
            4x4 robot pose transform, or None if input is None.

        Raises:
            ValueError: If ``camera_pose`` is not a finite 4x4 matrix.
        """
        if camera_pose is None:
            return None

        camera_pose_matrix = np.asarray(camera_pose, dtype=float)
        if camera_pose_matrix.shape != (4, 4) or not np.all(np.isfinite(camera_pose_matrix)):
            raise ValueError("camera_pose must be a finite 4x4 matrix")

        camera_from_robot_transform = self._build_inverse_transform()
        robot_pose = camera_pose_matrix @ camera_from_robot_transform
        return robot_pose

