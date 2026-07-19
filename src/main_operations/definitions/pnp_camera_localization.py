from pathlib import Path
from typing import List, Optional

import numpy as np
from pupil_apriltags import Detection

from ..modules.apriltags.pnp_localization import PnpLocalization
from ..modules.apriltags.utils.fmap_parser import load_fmap_file
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters
from src.utils.device_management_utils.compute_pool import ComputePool
from src.main_operations.definitions.base.base_class import OperationInstance
from src.webui.web_server import EagleEyeInterface


class PnpCameraLocalizationDefinition(OperationInstance):
    """Definition for camera localization operations using AprilTags."""

    def __init__(
        self,
        camera_bus_id: str,
        apriltag_map_path: str,
        camera_config_registry: CameraConfigRegistry | None = None,
        web_interface: EagleEyeInterface | None = None,
        compute_pool: ComputePool | None = None,
    ) -> None:
        """Initialize the camera localization definition.

        Args:
            camera_bus_id: Camera bus ID used to resolve calibration files.
            apriltag_map_path: Path to the apriltag map file.
            camera_config_registry: Injected shared camera config registry.
        """
        self.web_interface = web_interface
        self.compute_pool = compute_pool

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

    def run(self, detections: List[Detection]) -> Optional[np.ndarray]:
        """Estimate camera pose from AprilTag detections.

        Args:
            detections: List of AprilTag detection objects.

        Returns:
            4x4 transformation matrix representing camera pose in global coordinates,
            or None if pose estimation failed.
        """
        return self.pose_estimator.estimate_pose_from_detections(detections)
