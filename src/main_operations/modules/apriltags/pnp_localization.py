from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detection

from src.main_operations.modules.apriltags.utils.apriltag import Apriltag
from src.utils.colors import Colors


class PnpLocalization:
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
            camera_matrix: Camera intrinsic matrix.
            distortion_coefficients: Camera distortion coefficients.
            apriltag_map: Mapping of tag IDs to AprilTag objects.
        """
        self.camera_matrix = camera_matrix.astype(np.float32, copy=False)
        dist = distortion_coefficients.astype(np.float32, copy=False)
        if dist.ndim == 1 or (dist.ndim == 2 and dist.shape[0] == 1):
            dist = dist.reshape((-1, 1))
        self.distortion_coefficients = dist
        self.apriltag_map = apriltag_map

    @staticmethod
    def fast_se3_inverse(t: np.ndarray) -> np.ndarray:
        """Fast analytical inverse for SE(3) transformation matrix.

        Args:
            t: 4x4 SE(3) transformation matrix.

        Returns:
            4x4 inverse transformation matrix.
        """
        R = t[:3, :3]
        translation = t[:3, 3]

        r_inv = R.T
        t_inv = -r_inv @ translation

        t_inv_matrix = np.empty((4, 4), dtype=t.dtype)
        t_inv_matrix.fill(0)
        t_inv_matrix[3, 3] = 1
        t_inv_matrix[:3, :3] = r_inv
        t_inv_matrix[:3, 3] = t_inv

        return t_inv_matrix

    def _fast_rodrigues(self, rvec: np.ndarray) -> np.ndarray:
        """Compute rotation matrix from rotation vector using Rodrigues formula.

        Args:
            rvec: Rotation vector (3,) or (3, 1).

        Returns:
            Rotation matrix (3, 3) as float32.
        """
        v = rvec.reshape(3).astype(np.float32, copy=False)
        theta = float(np.linalg.norm(v))
        if np.isclose(theta, 0.0, rtol=1e-09, atol=1e-09):
            return np.eye(3, dtype=np.float32)
        n = v / theta
        nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
        K = np.array([[0.0, -nz, ny], [nz, 0.0, -nx], [-ny, nx, 0.0]], dtype=np.float32)
        s = np.float32(np.sin(theta))
        c = np.float32(np.cos(theta))
        K2 = K @ K
        R = np.eye(3, dtype=np.float32)
        R += s * K + (1.0 - c) * K2
        return R

    def _solution_quality(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rotation_vector: np.ndarray,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
        tag_count: int,
    ) -> List[float]:
        """Measure how much a solved pose can be trusted.

        The robot-side pose estimator needs standard deviations, and these three
        numbers are what it derives them from. Cost is one ``projectPoints`` call
        over at most a few dozen points, which is noise next to tag detection.

        Args:
            object_points: Stacked field-space tag corners, shape (4 * tags, 3).
            image_points: Matching image-space corners, shape (4 * tags, 2).
            rotation_vector: Rodrigues rotation from ``solvePnP``.
            rotation_matrix: Same rotation as a 3x3 matrix.
            translation_vector: Translation from ``solvePnP``.
            tag_count: Number of mapped tags that contributed to the solve.

        Returns:
            ``[tag_count, mean_tag_distance_m, mean_reprojection_error_px]``.
        """
        reprojected, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.distortion_coefficients,
        )
        reprojection_error = float(
            np.mean(
                np.linalg.norm(reprojected.reshape(-1, 2) - image_points, axis=1)
            )
        )

        tag_centers = object_points.reshape(-1, 4, 3).mean(axis=1)
        camera_space_centers = (
            tag_centers @ rotation_matrix.T + translation_vector.reshape(3)
        )
        mean_distance = float(
            np.mean(np.linalg.norm(camera_space_centers, axis=1))
        )

        return [float(tag_count), mean_distance, reprojection_error]

    def estimate_pose_from_detections(
        self,
        detections: List[Detection],
    ) -> Optional[Tuple[np.ndarray, List[float]]]:
        """Estimate camera pose from AprilTag detections.

        Args:
            detections: List of AprilTag detections.

        Returns:
            The 4x4 camera pose in field coordinates paired with its quality
            metrics ``[tag_count, mean_tag_distance_m, reprojection_error_px]``,
            or None if pose estimation failed.
        """
        image_points_list = []
        object_points_list = []
        valid_tags_found = 0

        for detection in detections:
            tag_id = detection.tag_id
            if tag_id not in self.apriltag_map:
                continue

            corners = detection.corners.astype(np.float32, copy=False)
            apriltag_obj = self.apriltag_map[tag_id]
            global_corners = apriltag_obj.global_corners.astype(np.float32, copy=False)

            if not np.all(np.isfinite(corners)) or not np.all(
                np.isfinite(global_corners)
            ):
                continue

            image_points_list.append(corners)
            object_points_list.append(global_corners)
            valid_tags_found += 1

        if valid_tags_found == 0:
            return None

        image_points = np.vstack(image_points_list).astype(np.float32, copy=False)
        object_points = np.vstack(object_points_list).astype(np.float32, copy=False)

        if not (
            np.all(np.isfinite(image_points)) and np.all(np.isfinite(object_points))
        ):
            return None

        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if (
            not success
            or rotation_vector is None
            or translation_vector is None
            or not np.all(np.isfinite(rotation_vector))
            or not np.all(np.isfinite(translation_vector))
        ):
            return None

        rotation_matrix = self._fast_rodrigues(rotation_vector)
        camera_space_transform = np.eye(4, dtype=np.float32)
        camera_space_transform[:3, :3] = rotation_matrix
        camera_space_transform[:3, 3] = translation_vector.flatten().astype(
            np.float32, copy=False
        )

        global_camera_transform = PnpLocalization.fast_se3_inverse(
            camera_space_transform
        )

        if not np.all(np.isfinite(global_camera_transform)):
            print(
                f"{Colors.YELLOW}Operation - Skipping publish of robot transform due to non-finite values{Colors.RESET}"
            )
            return None

        quality = self._solution_quality(
            object_points,
            image_points,
            rotation_vector,
            rotation_matrix,
            translation_vector,
            valid_tags_found,
        )
        return global_camera_transform, quality
