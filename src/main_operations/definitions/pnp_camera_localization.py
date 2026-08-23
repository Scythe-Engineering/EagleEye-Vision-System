import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pupil_apriltags import Detection

from ..modules.apriltags.pnp_localization import PnpLocalization
from ..modules.apriltags.utils.fmap_parser import load_fmap_file
from src.secondary_operations.camera_to_robot_pose import (
    build_robot_from_camera_transform,
)
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters
from src.main_operations.definitions.base.base_class import OperationInstance
from src.webui.web_server import EagleEyeInterface

# The robot-frame matrices in this pipeline use OpenCV-style EDN body axes
# (x right, y down, z forward); publish_to_networktables converts them to
# WPILib NWU with the transpose of this matrix. Robot yaw arrives in WPILib
# convention (radians, CCW positive about field +z), so the constrained
# rotation is built in NWU and mapped back through this basis.
_NWU_FROM_EDN = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


class PnpCameraLocalizationDefinition(OperationInstance):
    """Definition for camera localization operations using AprilTags."""

    def __init__(
        self,
        camera_bus_id: str,
        apriltag_map_path: str,
        camera_config_registry: CameraConfigRegistry | None = None,
        web_interface: EagleEyeInterface | None = None,
    ) -> None:
        """Initialize the camera localization definition.

        Args:
            camera_bus_id: Camera bus ID used to resolve calibration files.
            apriltag_map_path: Path to the apriltag map file.
            camera_config_registry: Injected shared camera config registry.
        """
        self.web_interface = web_interface
        self.camera_bus_id = str(camera_bus_id)
        self.camera_config_registry = camera_config_registry

        intrinsics_path: str
        if camera_config_registry is not None:
            camera_config = camera_config_registry.get_config(camera_bus_id)
            if camera_config.intrinsics_path is None:
                raise ValueError(
                    f"No intrinsics path found for camera bus ID '{camera_bus_id}'"
                )
            intrinsics_path = camera_config.intrinsics_path
        else:
            intrinsics_path = str(
                Path(__file__).resolve().parents[2]
                / "utils"
                / "camera_utils"
                / "camera_calibrations"
                / camera_bus_id
                / "intrinsics.json"
            )

        camera_matrix, distortion_coefficients = load_camera_parameters(
            intrinsics_path
        )
        apriltag_map = load_fmap_file(apriltag_map_path)

        self.pose_estimator = PnpLocalization(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            apriltag_map=apriltag_map,
        )

    def _field_from_camera_rotation(self, robot_yaw: float) -> np.ndarray:
        """Build the fixed camera rotation implied by a known robot yaw.

        Assumes the robot sits flat on the field, so the gyro yaw plus the
        camera mounting extrinsics fully determine the camera's orientation.

        Args:
            robot_yaw: Robot heading in radians, WPILib convention.

        Returns:
            3x3 ``R_field_from_camera`` for the constrained solve.
        """
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        field_from_robot_nwu = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        robot_from_camera = build_robot_from_camera_transform(
            self.camera_config_registry, self.camera_bus_id
        )[:3, :3]
        return field_from_robot_nwu @ _NWU_FROM_EDN @ robot_from_camera

    def run(self, input_data: Any) -> Dict[str, Any]:
        """Estimate camera pose from AprilTag detections.

        Args:
            input_data: AprilTag detection objects, either bare or as a dict
                with ``detections`` and an optional ``robot_yaw`` (radians,
                WPILib convention, usually from ``get_networktables_value``).
                When a finite yaw is present the solve is yaw-constrained.

        Returns:
            Mapping of ``camera_pose`` to a 4x4 transform in global coordinates and
            ``pose_meta`` to ``[tag_count, mean_tag_distance_m, reprojection_error_px]``.
            Both ports carry None when pose estimation failed, so downstream
            operations keep their existing None handling.
        """
        if isinstance(input_data, dict):
            detections: List[Detection] = input_data.get("detections")
            robot_yaw = input_data.get("robot_yaw")
        else:
            detections = input_data
            robot_yaw = None

        fixed_rotation: np.ndarray | None = None
        if robot_yaw is not None and math.isfinite(float(robot_yaw)):
            fixed_rotation = self._field_from_camera_rotation(float(robot_yaw))

        solution = self.pose_estimator.estimate_pose_from_detections(
            detections, field_from_camera_rotation=fixed_rotation
        )
        if solution is None:
            return {"camera_pose": None, "pose_meta": None}
        camera_pose, pose_meta = solution
        return {"camera_pose": camera_pose, "pose_meta": pose_meta}
