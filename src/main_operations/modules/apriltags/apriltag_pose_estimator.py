from typing import Dict, Optional, List

import cv2
import numpy as np
from pupil_apriltags import Detection

from src.main_operations.modules.apriltags.utils.apriltag import Apriltag


class AprilTagPoseEstimator:
    """Class for AprilTag pose estimation from detection results.

    This class handles pose estimation from AprilTag detections without
    performing the detection itself, allowing for separation of concerns.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        apriltag_map: Dict[int, Apriltag],
    ) -> None:
        """Initialize the AprilTag pose estimator.

        Args:
            camera_matrix (np.ndarray): Camera intrinsic matrix.
            distortion_coefficients (np.ndarray): Camera distortion coefficients.
            apriltag_map (Dict[int, Apriltag]): Mapping of tag IDs to AprilTag objects.
        """
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.apriltag_map = apriltag_map

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image using the camera matrix and distortion coefficients.

        Args:
            image (np.ndarray): Input image (BGR format).

        Returns:
            np.ndarray: Undistorted image.
        """
        return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients)

    def estimate_pose_from_detections(
        self,
        detections: List[Detection],
    ) -> Optional[np.ndarray]:
        """Estimate camera pose from AprilTag detections.

        Args:
            detections (List[Detection]): List of AprilTag detections.

        Returns:
            Optional[np.ndarray]: 4x4 transformation matrix representing camera pose,
                                or None if pose estimation failed.
        """
        image_points_list = []
        object_points_list = []
        valid_tags_found = False

        for detection in detections:
            tag_id = detection.tag_id
            if tag_id not in self.apriltag_map:
                continue

            image_points_list.extend(detection.corners.astype(np.float32))
            apriltag_obj = self.apriltag_map[tag_id]
            object_points_list.extend(apriltag_obj.global_corners.astype(np.float32))
            valid_tags_found = True

        if not valid_tags_found or len(image_points_list) < 4:
            return None

        image_points = np.array(image_points_list, dtype=np.float32).reshape(-1, 2)
        object_points = np.array(object_points_list, dtype=np.float32).reshape(-1, 3)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.distortion_coefficients,
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        camera_space_transform = np.eye(4)
        camera_space_transform[:3, :3] = rotation_matrix
        camera_space_transform[:3, 3] = translation_vector.flatten()

        global_camera_transform = np.linalg.inv(camera_space_transform)
        return global_camera_transform
