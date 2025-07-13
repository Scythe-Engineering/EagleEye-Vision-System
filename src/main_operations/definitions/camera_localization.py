from typing import List, Optional

import numpy as np
from pupil_apriltags import Detection

from ..modules.apriltags.apriltag_pose_estimator import AprilTagPoseEstimator
from ..modules.apriltags.utils.fmap_parser import load_fmap_file
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters


class CameraLocalizationDefinition:
    """Definition for camera localization operations using AprilTags."""
    
    def __init__(
        self,
        camera_parameters_path: str,
        apriltag_map_path: str
    ) -> None:
        """Initialize the camera localization definition.

        Args:
            camera_parameters_path: Path to the camera parameters file.
            apriltag_map_path: Path to the apriltag map file.
        """
        camera_matrix, distortion_coefficients = load_camera_parameters(camera_parameters_path)
        apriltag_map = load_fmap_file(apriltag_map_path)
        
        self.pose_estimator = AprilTagPoseEstimator(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            apriltag_map=apriltag_map
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
