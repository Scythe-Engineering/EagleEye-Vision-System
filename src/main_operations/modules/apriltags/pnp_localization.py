from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
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
        refinement_iterations: int = 10,
    ) -> None:
        """Initialize the AprilTag pose estimator.

        Args:
            camera_matrix: Camera intrinsic matrix.
            distortion_coefficients: Camera distortion coefficients.
            apriltag_map: Mapping of tag IDs to AprilTag objects.
            refinement_iterations: Maximum LM refinement iterations; zero disables refinement.
        """
        self.camera_matrix = camera_matrix.astype(np.float64, copy=False)
        dist = distortion_coefficients.astype(np.float64, copy=False)
        if dist.ndim == 1 or (dist.ndim == 2 and dist.shape[0] == 1):
            dist = dist.reshape((-1, 1))
        self.distortion_coefficients = dist
        self.apriltag_map = apriltag_map
        self.set_refinement_iterations(refinement_iterations)

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

    def set_refinement_iterations(self, iterations: int) -> None:
        """Set the bounded, live-updatable LM iteration limit."""
        value = int(iterations)
        if (
            isinstance(iterations, bool)
            or value != float(iterations)
            or not 0 <= value <= 100
        ):
            raise ValueError("refinement_iterations must be an integer from 0 to 100")
        self.refinement_iterations = value

    def _solve(
        self, object_points: np.ndarray, image_points: np.ndarray, tag_count: int
    ):
        """Initialize with planar IPPE or multi-tag SQPnP, then refine valid candidates.

        Candidate ranking uses squared reprojection error. Both initialization and
        refinement are stateless: a previous pose never masks motion or tag ambiguity.
        IPPE accepts arbitrary coplanar field points, avoiding IPPE_SQUARE's special
        local corner order. No field/camera basis conversion occurs in this solver.
        """
        if np.linalg.matrix_rank(object_points - object_points.mean(axis=0)) < 2:
            return None
        iterations = self.refinement_iterations
        try:
            solved = cv2.solvePnPGeneric(
                object_points,
                image_points,
                self.camera_matrix,
                self.distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE if tag_count == 1 else cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            return None
        if not solved[0]:
            return None
        rotations, translations = list(solved[1]), list(solved[2])
        # Coplanar multi-tag layouts can have two plausible minima. SQPnP alone
        # need not expose both; evaluate IPPE's planar hypotheses as well.
        singular_values = np.linalg.svd(
            object_points - object_points.mean(axis=0), compute_uv=False
        )
        if tag_count > 1 and singular_values[-1] < 1e-6 * singular_values[0]:
            try:
                planar = cv2.solvePnPGeneric(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.distortion_coefficients,
                    flags=cv2.SOLVEPNP_IPPE,
                )
                if planar[0]:
                    rotations.extend(planar[1])
                    translations.extend(planar[2])
            except cv2.error:
                pass

        def score(rvec, tvec):
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                return float("inf")
            rotation = cv2.Rodrigues(rvec)[0]
            if np.any((object_points @ rotation.T + tvec.reshape(3))[:, 2] <= 0):
                return float("inf")
            projected = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                self.camera_matrix,
                self.distortion_coefficients,
            )[0]
            return float(
                np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))
            )

        best = None
        best_error = float("inf")
        for rotation, translation in zip(rotations, translations):
            error = score(rotation, translation)
            if not np.isfinite(error):
                continue
            if iterations:
                try:
                    refined_rotation, refined_translation = cv2.solvePnPRefineLM(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.distortion_coefficients,
                        rotation.copy(),
                        translation.copy(),
                        criteria=(
                            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                            iterations,
                            1e-9,
                        ),
                    )
                    refined_error = score(refined_rotation, refined_translation)
                    if refined_error <= error:
                        rotation, translation, error = (
                            refined_rotation,
                            refined_translation,
                            refined_error,
                        )
                except cv2.error:
                    pass  # A valid initializer is preferable to dropping this frame.
            if error < best_error:
                best = rotation, translation
                best_error = error
        return best

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
            np.mean(np.linalg.norm(reprojected.reshape(-1, 2) - image_points, axis=1))
        )

        tag_centers = object_points.reshape(-1, 4, 3).mean(axis=1)
        camera_space_centers = (
            tag_centers @ rotation_matrix.T + translation_vector.reshape(3)
        )
        mean_distance = float(np.mean(np.linalg.norm(camera_space_centers, axis=1)))

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
        seen_ids = set()

        for detection in detections:
            tag_id = detection.tag_id
            if tag_id not in self.apriltag_map or tag_id in seen_ids:
                continue

            corners = detection.corners.astype(np.float64, copy=False)
            apriltag_obj = self.apriltag_map[tag_id]
            global_corners = apriltag_obj.global_corners.astype(np.float64, copy=False)

            if corners.shape != (4, 2) or global_corners.shape != (4, 3):
                continue
            if not np.all(np.isfinite(corners)) or not np.all(
                np.isfinite(global_corners)
            ):
                continue

            seen_ids.add(tag_id)
            image_points_list.append(corners)
            object_points_list.append(global_corners)
            valid_tags_found += 1

        if valid_tags_found == 0:
            return None

        image_points = np.vstack(image_points_list).astype(np.float64, copy=False)
        object_points = np.vstack(object_points_list).astype(np.float64, copy=False)

        if not (
            np.all(np.isfinite(image_points)) and np.all(np.isfinite(object_points))
        ):
            return None

        # Optimize around nearby tag geometry rather than the distant field origin.
        # This improves rotation/translation conditioning, particularly in older
        # OpenCV LM builds, without changing the published field coordinates.
        object_origin = object_points.mean(axis=0)
        solution = self._solve(
            object_points - object_origin, image_points, valid_tags_found
        )
        if solution is None:
            return None
        rotation_vector, translation_vector = solution
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        translation_vector = translation_vector.reshape(3, 1) - (
            rotation_matrix @ object_origin
        ).reshape(3, 1)
        camera_space_transform = np.eye(4, dtype=np.float64)
        camera_space_transform[:3, :3] = rotation_matrix
        camera_space_transform[:3, 3] = translation_vector.reshape(3)
        global_camera_transform = self.fast_se3_inverse(camera_space_transform)
        if not np.all(np.isfinite(global_camera_transform)):
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
