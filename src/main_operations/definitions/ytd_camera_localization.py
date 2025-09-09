from __future__ import annotations

from typing import Optional, List, TypedDict

from src.main_operations.modules.apriltags.ytd_camera_localization.ytd_localization import (
    YtdLocalization,
)
from src.main_operations.modules.apriltags.utils.fmap_parser import load_fmap_file
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters
import numpy as np


class ApriltagDetection(TypedDict):
    """Type definition for AprilTag detection data."""
    tag_id: int
    corners: np.ndarray
    center: np.ndarray
    pose_R: Optional[np.ndarray]
    pose_t: Optional[np.ndarray]
    tag_family: str
    decision_margin: float
    hamming: int
    homography: np.ndarray


class LocalizationResult(TypedDict):
    """Type definition for camera localization result."""
    transform: np.ndarray


class YtdCameraLocalizationDefinition:
    """Wrapper that delegates to the `YtdLocalization` implementation.

    Constructor parameters mirror the implementation and are provided via action params.
    """

    def __init__(
        self,
        camera_parameters_path: str,
        apriltag_map_path: str,
    ) -> None:
        """Initialize the localization definition.

        Args:
                camera_parameters_path: Path to the camera parameters file.
                apriltag_map_path: Path to the AprilTag map (fmap) file.
        """
        camera_matrix, distortion_coefficients = load_camera_parameters(
            camera_parameters_path
        )
        apriltag_map = load_fmap_file(apriltag_map_path)

        self.pose_estimator = YtdLocalization(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            apriltag_map=apriltag_map,
        )

    def set_attribute(self, attribute_name: str, value: float) -> None:
        """Set a named attribute on the underlying implementation.

        Args:
                attribute_name: Name of the attribute to set. Supported: "camera_yaw", "camera_pitch".
                value: Float value to assign to the attribute.
        """
        self.pose_estimator.set_attribute(attribute_name, value)

    def run(self, detections: List[ApriltagDetection]) -> Optional[np.ndarray]:
        """Delegate run to the `YtdLocalization` implementation.

        Args:
                detections: List of AprilTag detections.

        Returns:
                4x4 transformation matrix as numpy array or None if localization fails.
        """
        return self.pose_estimator.estimate_pose_from_detections(detections)

    def visualize(self, frame: np.ndarray) -> None:
        """Visualize the YTD camera localization outputs.

        This operation returns pose estimation data (transform/numbers) only,
        so no frame visualization is available.

        Args:
            frame: Input frame (unused).

        Returns:
            None - no visualization available for transform-only operations.
        """
        return None

    def update_config(self, _: dict) -> None:
        """Update the configuration of the YTD camera localization. No live-updatable parameters available.

        Args:
            json_config: JSON configuration for the YTD camera localization (ignored).
        """
        pass
