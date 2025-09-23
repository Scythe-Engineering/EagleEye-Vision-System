from typing import Dict, List, Optional

import cv2
import numpy as np
from line_profiler import profile
from pupil_apriltags import Detection

from src.main_operations.modules.apriltags.utils.apriltag import Apriltag


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
            camera_matrix (np.ndarray): Camera intrinsic matrix.
            distortion_coefficients (np.ndarray): Camera distortion coefficients.
            apriltag_map (Dict[int, Apriltag]): Mapping of tag IDs to AprilTag objects.
        """
        self.camera_matrix = camera_matrix.astype(np.float32, copy=False)
        self.distortion_coefficients = distortion_coefficients.astype(
            np.float32, copy=False
        )
        self.apriltag_map = apriltag_map
        self.last_pose = None
        self._last_camera_space_pose = None
        # Cache last rvec to avoid computing it from R for iterative guesses
        self._last_rvec = None

    @profile
    def fast_se3_inverse(self, t: np.ndarray) -> np.ndarray:
        """Fast analytical inverse for SE(3) transformation matrix.

        Args:
            t (np.ndarray): 4x4 SE(3) transformation matrix.

        Returns:
            np.ndarray: 4x4 inverse transformation matrix.
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

    @profile
    def _fast_rodrigues(self, rvec: np.ndarray) -> np.ndarray:
        """Compute rotation matrix from rotation vector using Rodrigues formula in NumPy.

        Args:
            rvec (np.ndarray): Rotation vector (3, ) or (3, 1).

        Returns:
            np.ndarray: Rotation matrix (3, 3) as float32.
        """
        v = rvec.reshape(3).astype(np.float32, copy=False)
        theta = float(np.linalg.norm(v))
        if theta == 0.0:
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

    @profile
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
        valid_tags_found = 0

        for detection in detections:
            tag_id = detection.tag_id
            if tag_id not in self.apriltag_map:
                continue

            # Ensure float32 for OpenCV performance
            image_points_list.append(detection.corners.astype(np.float32, copy=False))
            apriltag_obj = self.apriltag_map[tag_id]
            object_points_list.append(
                apriltag_obj.global_corners.astype(np.float32, copy=False)
            )
            valid_tags_found += 1

        if valid_tags_found == 0:
            return None

        # Avoid extra copies for single-tag case
        if valid_tags_found == 1:
            image_points = image_points_list[0]
            object_points = object_points_list[0]
        else:
            image_points = np.vstack(image_points_list).astype(np.float32, copy=False)
            object_points = np.vstack(object_points_list).astype(np.float32, copy=False)

        camera_space_transform = None

        if valid_tags_found == 1:
            # Single tag: Use IPPE_SQUARE with solvePnPGeneric for multiple solutions
            retval, rotation_vectors, translation_vectors, reprojection_error = (
                cv2.solvePnPGeneric(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.distortion_coefficients,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            )

            if retval and len(rotation_vectors) > 0:
                # Choose the solution closest to the last camera-space pose using translation only.
                last_cam_pose = self._last_camera_space_pose
                if last_cam_pose is not None:
                    last_translation = last_cam_pose[:3, 3]
                    best_index = 0
                    best_distance = float("inf")
                    for i in range(len(translation_vectors)):
                        t_vec = translation_vectors[i].reshape(3)
                        distance = float(np.linalg.norm(t_vec - last_translation))
                        if distance < best_distance:
                            best_distance = distance
                            best_index = i
                else:
                    best_index = 0

                # Compute rotation matrix only once for the selected candidate
                rotation_matrix = self._fast_rodrigues(rotation_vectors[best_index])
                camera_space_transform = np.eye(4, dtype=np.float32)
                camera_space_transform[:3, :3] = rotation_matrix
                camera_space_transform[:3, 3] = (
                    translation_vectors[best_index]
                    .flatten()
                    .astype(np.float32, copy=False)
                )
                # Cache rvec from solver directly
                self._last_rvec = rotation_vectors[best_index].astype(
                    np.float32, copy=False
                )
        else:
            # Multiple tags: Prefer iterative with extrinsic guess if available for speed/stability
            if self._last_camera_space_pose is not None and self._last_rvec is not None:
                last_t = self._last_camera_space_pose[:3, 3]
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.distortion_coefficients,
                    rvec=self._last_rvec,
                    tvec=last_t.reshape(3, 1).astype(np.float32, copy=False),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.distortion_coefficients,
                    flags=cv2.SOLVEPNP_SQPNP,
                )

            if success:
                rotation_matrix = self._fast_rodrigues(rotation_vector)
                camera_space_transform = np.eye(4, dtype=np.float32)
                camera_space_transform[:3, :3] = rotation_matrix
                camera_space_transform[:3, 3] = translation_vector.flatten().astype(
                    np.float32, copy=False
                )
                # Cache rvec from solver directly
                self._last_rvec = rotation_vector.astype(np.float32, copy=False)

        if camera_space_transform is None:
            return None

        global_camera_transform = self.fast_se3_inverse(camera_space_transform)
        # Cache both spaces for future calls to avoid extra inversions
        self._last_camera_space_pose = camera_space_transform
        self.last_pose = global_camera_transform
        return global_camera_transform
