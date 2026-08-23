from __future__ import annotations

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.quaternion_utils import euler_to_rotation_matrix


def build_robot_from_camera_transform(
    camera_config_registry: CameraConfigRegistry | None,
    camera_bus_id: str,
) -> np.ndarray:
    """Build ``T_robot_from_camera`` from a camera's registered extrinsics.

    This is the one definition of the pipeline's camera-mounting transform;
    every operation that needs to relate the camera frame to the robot frame
    must use it so the conventions stay consistent.

    Args:
        camera_config_registry: Shared camera config registry, or None.
        camera_bus_id: Camera bus ID used to resolve extrinsics.

    Returns:
        4x4 transform mapping points from camera frame to robot frame, identity
        when no registry is available.
    """
    if camera_config_registry is None:
        return np.eye(4, dtype=float)

    camera_config = camera_config_registry.get_config(camera_bus_id)
    extrinsics = camera_config.extrinsics

    # stuff in strange order because of coordinate system conversion for both frontend and wpilib/robot
    transform = euler_to_rotation_matrix(
        pitch=float(-extrinsics.yaw),
        yaw=float(-extrinsics.pitch),
        roll=float(extrinsics.roll),
    )
    transform[:3, 3] = np.array(
        [
            float(extrinsics.y_offset),
            float(-extrinsics.z_offset),
            float(extrinsics.x_offset),
        ],
        dtype=float,
    )
    return transform


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
        self._cached_camera_from_robot_transform: np.ndarray | None = None

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

    def _build_inverse_transform(self) -> np.ndarray:
        """Build ``T_camera_from_robot`` by inverting camera extrinsics.

        Returns:
            4x4 transform mapping robot frame to camera frame.
        """
        robot_from_camera = build_robot_from_camera_transform(
            self.camera_config_registry, self.camera_bus_id
        )
        return self._fast_se3_inverse(robot_from_camera)

    def _get_cached_inverse_transform(self) -> np.ndarray:
        """Get cached ``T_camera_from_robot`` and build it lazily if needed.

        Returns:
            Cached or newly built 4x4 inverse extrinsics transform.
        """
        if self._cached_camera_from_robot_transform is None:
            self._cached_camera_from_robot_transform = self._build_inverse_transform()
        return self._cached_camera_from_robot_transform

    def update_config(self, json_config: dict[str, object]) -> None:
        """Update runtime-configurable operation parameters.

        Args:
            json_config: Configuration dictionary with updated values.
        """
        invalidate_cache = False

        if "camera_bus_id" in json_config:
            next_camera_bus_id = str(json_config["camera_bus_id"])
            if next_camera_bus_id != self.camera_bus_id:
                self.camera_bus_id = next_camera_bus_id
                invalidate_cache = True

        extrinsics_keys = {
            "pitch",
            "yaw",
            "roll",
            "x_offset",
            "y_offset",
            "z_offset",
        }
        if any(key in json_config for key in extrinsics_keys):
            invalidate_cache = True

        if invalidate_cache:
            self._cached_camera_from_robot_transform = None

    def run(self, camera_pose: np.ndarray | None) -> np.ndarray | None:
        """Convert camera pose to robot pose.

        Args:
            camera_pose: 4x4 camera pose transform, or None.

        Returns:
            4x4 robot pose transform, or None if input is None/invalid.
        """
        if camera_pose is None:
            return None

        camera_pose_matrix = np.asarray(camera_pose, dtype=float)
        if camera_pose_matrix.shape != (4, 4) or not np.all(np.isfinite(camera_pose_matrix)):
            return None

        camera_from_robot_transform = self._get_cached_inverse_transform()
        robot_pose = camera_pose_matrix @ camera_from_robot_transform
        return robot_pose
