from typing import Dict, List, Optional
import cv2
import numpy as np
from line_profiler import profile
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
            camera_matrix (np.ndarray): Camera intrinsic matrix.
            distortion_coefficients (np.ndarray): Camera distortion coefficients.
            apriltag_map (Dict[int, Apriltag]): Mapping of tag IDs to AprilTag objects.
        """
        self.camera_matrix = camera_matrix.astype(np.float32, copy=False)
        dist = distortion_coefficients.astype(np.float32, copy=False)
        if dist.ndim == 1 or (dist.ndim == 2 and dist.shape[0] == 1):
            dist = dist.reshape((-1, 1))
        self.distortion_coefficients = dist

        self.apriltag_map = apriltag_map
        self.last_pose = None
        self._last_camera_space_pose = None
        self._last_rvec = None
        self.non_finite_count = 0

    def clear_position_cache(self) -> None:
        """Clear the position cache and reset non-finite counter."""
        self.last_pose = None
        self._last_camera_space_pose = None
        self._last_rvec = None
        self.non_finite_count = 0

    @staticmethod
    def fast_se3_inverse(t: np.ndarray) -> np.ndarray:
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

            corners = detection.corners.astype(np.float32, copy=False)
            if not np.all(np.isfinite(corners)):
                continue
            image_points_list.append(corners)
            apriltag_obj = self.apriltag_map[tag_id]
            global_corners = apriltag_obj.global_corners.astype(np.float32, copy=False)
            if not np.all(np.isfinite(global_corners)):
                continue
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

        success = False
        rotation_vector = None
        translation_vector = None

        # First attempt with extrinsic guess if available
        use_guess = (
            self._last_camera_space_pose is not None
            and self._last_rvec is not None
            and np.all(np.isfinite(self._last_camera_space_pose))
            and np.all(np.isfinite(self._last_rvec))
        )

        if use_guess:
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
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        camera_space_transform = None
        if success and rotation_vector is not None and translation_vector is not None:
            if not (
                np.all(np.isfinite(rotation_vector))
                and np.all(np.isfinite(translation_vector))
            ):
                return None
            rotation_matrix = self._fast_rodrigues(rotation_vector)
            camera_space_transform = np.eye(4, dtype=np.float32)
            camera_space_transform[:3, :3] = rotation_matrix
            camera_space_transform[:3, 3] = translation_vector.flatten().astype(
                np.float32, copy=False
            )
            if np.all(np.isfinite(rotation_vector)):
                self._last_rvec = rotation_vector.astype(np.float32, copy=False)

        if camera_space_transform is None:
            return None

        global_camera_transform = PnpLocalization.fast_se3_inverse(
            camera_space_transform
        )

        # Check if pose is far from previous pose (> 2m)
        if (
            self.last_pose is not None
            and np.all(np.isfinite(global_camera_transform))
            and use_guess
        ):
            distance = np.linalg.norm(
                global_camera_transform[:3, 3] - self.last_pose[:3, 3]
            )
            if distance > 2.0:
                # Clear cache and recompute without extrinsic guess
                self.clear_position_cache()

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.distortion_coefficients,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

                if (
                    success
                    and rotation_vector is not None
                    and translation_vector is not None
                ):
                    if not (
                        np.all(np.isfinite(rotation_vector))
                        and np.all(np.isfinite(translation_vector))
                    ):
                        return None
                    rotation_matrix = self._fast_rodrigues(rotation_vector)
                    camera_space_transform = np.eye(4, dtype=np.float32)
                    camera_space_transform[:3, :3] = rotation_matrix
                    camera_space_transform[:3, 3] = translation_vector.flatten().astype(
                        np.float32, copy=False
                    )
                    if np.all(np.isfinite(rotation_vector)):
                        self._last_rvec = rotation_vector.astype(np.float32, copy=False)

                    global_camera_transform = PnpLocalization.fast_se3_inverse(
                        camera_space_transform
                    )

        if np.all(np.isfinite(camera_space_transform)):
            self._last_camera_space_pose = camera_space_transform
        self.last_pose = global_camera_transform

        if not np.all(np.isfinite(global_camera_transform)):
            print(
                f"{Colors.YELLOW}Operation - Skipping publish of robot transform due to non-finite values{Colors.RESET}"
            )
            self.non_finite_count += 1
            if self.non_finite_count >= 3:
                self.clear_position_cache()
                print(
                    f"{Colors.CYAN}Position cache cleared due to 3 consecutive non-finite values{Colors.RESET}"
                )
            return None

        # Reset counter on successful finite result
        self.non_finite_count = 0
        return global_camera_transform
