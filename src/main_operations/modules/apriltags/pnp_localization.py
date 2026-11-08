import logging
import math
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detection

from src.main_operations.modules.apriltags.utils.apriltag import Apriltag

# Optional native LM refiner (built automatically); OpenCV's refiner is the fallback.
try:
    from pnp_refiner import PnpRefiner as RustPnpRefiner  # type: ignore
except ImportError:
    RustPnpRefiner = None

logger = logging.getLogger(__name__)

# Stop LM refinement once a step's norm falls below this.
REFINEMENT_EPSILON = 1e-9

# Distinct ordered tag-id tuples seen recently; a field yields far fewer in practice.
TAG_SET_CACHE_SIZE = 256


class TagSetGeometry(NamedTuple):
    """Field geometry that depends only on which tags were seen, in detection order."""

    field_points: np.ndarray
    """Stacked field-space corners, shape (4 * tags, 3)."""
    origin: np.ndarray
    """Mean of ``field_points``; solving around it improves conditioning."""
    object_points: np.ndarray
    """``field_points - origin``."""
    tag_centers: np.ndarray
    """Field-space center of each tag, shape (tags, 3)."""
    tag_count: int
    is_degenerate: bool
    """Fewer than two independent directions, so no pose is recoverable."""
    is_coplanar: bool
    """All corners lie in one plane, so IPPE's planar hypotheses also apply."""


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
        self._field_corners = self._usable_field_corners(apriltag_map)
        self._tag_set_cache: OrderedDict[tuple, TagSetGeometry] = OrderedDict()
        self._rust_refiner = self._create_rust_refiner()
        self.set_refinement_iterations(refinement_iterations)

    def _create_rust_refiner(self):
        """Return the native LM refiner, or None to use OpenCV's.

        The native refiner supports 0, 4, 5 or 8 distortion coefficients; other
        models, or a missing build, keep OpenCV's ``solvePnPRefineLM``.
        """
        if RustPnpRefiner is None:
            return None
        try:
            return RustPnpRefiner(
                np.ascontiguousarray(self.camera_matrix),
                np.ascontiguousarray(self.distortion_coefficients.ravel()),
            )
        except ValueError as exc:
            logger.info("Using OpenCV LM refinement: %s", exc)
            return None

    def _refine(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run at most ``refinement_iterations`` LM steps from a copy of a pose.

        Raises:
            cv2.error: If OpenCV's refiner rejects the inputs.
        """
        if self._rust_refiner is not None:
            refined_rotation = rotation.astype(np.float64)
            refined_translation = translation.astype(np.float64)
            if self._rust_refiner.refine(
                object_points,
                image_points,
                refined_rotation,
                refined_translation,
                self.refinement_iterations,
                REFINEMENT_EPSILON,
            ):
                return refined_rotation, refined_translation
        return cv2.solvePnPRefineLM(
            object_points,
            image_points,
            self.camera_matrix,
            self.distortion_coefficients,
            rotation.copy(),
            translation.copy(),
            criteria=(
                cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                self.refinement_iterations,
                REFINEMENT_EPSILON,
            ),
        )

    @staticmethod
    def _usable_field_corners(
        apriltag_map: dict[int, Apriltag],
    ) -> dict[int, np.ndarray]:
        """Convert each mapped tag's corners to float64 once, dropping unusable tags."""
        usable = {}
        for tag_id, apriltag_obj in apriltag_map.items():
            corners = apriltag_obj.global_corners.astype(np.float64, copy=False)
            if corners.shape == (4, 3) and np.isfinite(corners).all():
                usable[tag_id] = corners
        return usable

    def _tag_set_geometry(self, tag_ids: tuple) -> TagSetGeometry:
        """Return cached geometry for an ordered tag set, computing it on a miss."""
        geometry = self._tag_set_cache.get(tag_ids)
        if geometry is not None:
            self._tag_set_cache.move_to_end(tag_ids)
            return geometry

        field_points = np.vstack([self._field_corners[tag_id] for tag_id in tag_ids])
        origin = field_points.mean(axis=0)
        # Map corners can be Fortran-ordered; the native refiner needs C order.
        object_points = np.ascontiguousarray(field_points - origin)
        centered_points = object_points - object_points.mean(axis=0)
        singular_values = np.linalg.svd(centered_points, compute_uv=False)
        rank_tolerance = (
            singular_values[0]
            * max(centered_points.shape)
            * np.finfo(centered_points.dtype).eps
        )
        geometry = TagSetGeometry(
            field_points=field_points,
            origin=origin,
            object_points=object_points,
            tag_centers=field_points.reshape(-1, 4, 3).mean(axis=1),
            tag_count=len(tag_ids),
            is_degenerate=bool(np.count_nonzero(singular_values > rank_tolerance) < 2),
            is_coplanar=bool(singular_values[-1] < 1e-6 * singular_values[0]),
        )
        self._tag_set_cache[tag_ids] = geometry
        if len(self._tag_set_cache) > TAG_SET_CACHE_SIZE:
            self._tag_set_cache.popitem(last=False)
        return geometry

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
        """Set the bounded, live-updatable LM iteration limit.

        Args:
            iterations: Maximum LM refinement iterations, from 0 to 100.

        Raises:
            ValueError: If iterations is not an integer from 0 to 100.
        """
        value = int(iterations)
        if (
            isinstance(iterations, bool)
            or value != float(iterations)
            or not 0 <= value <= 100
        ):
            raise ValueError("refinement_iterations must be an integer from 0 to 100")
        self.refinement_iterations = value

    def _solve(
        self,
        geometry: TagSetGeometry,
        image_points: np.ndarray,
        previous_pose: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
        """Initialize with planar IPPE or multi-tag SQPnP, then refine valid candidates.

        Candidate ranking uses squared reprojection error. An optional previous field
        pose only breaks near-ties between bounded single-tag IPPE hypotheses.
        IPPE accepts arbitrary coplanar field points, avoiding IPPE_SQUARE's special
        local corner order. No field/camera basis conversion occurs in this solver.

        Args:
            geometry: Tag-set geometry; rotation and translation are solved for its
                mean-centered ``object_points``.
            image_points: Corresponding image corners in pixels, shape (4 * tags, 2).
            previous_pose: Recent field-from-camera pose used only for single-tag ties.

        Returns:
            Best rotation and translation vectors with the squared per-corner pixel
            distances measured while scoring them, or None if no candidate is valid.
            The distances are None when the winner was scored from OpenCV's RMS.
        """
        if geometry.is_degenerate:
            return None
        object_points = geometry.object_points
        object_origin = geometry.origin
        tag_count = geometry.tag_count
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
        initial_rms = list(solved[3].ravel())
        # Coplanar multi-tag layouts can have two plausible minima. SQPnP alone
        # need not expose both; evaluate IPPE's planar hypotheses as well.
        if tag_count > 1 and geometry.is_coplanar:
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
                    initial_rms.extend(planar[3].ravel())
            except cv2.error as exc:
                logger.error("Planar IPPE candidate generation failed: %s", exc)

        def score(
            rvec: np.ndarray, tvec: np.ndarray, coordinate_rms: float | None = None
        ) -> tuple[float, np.ndarray | None]:
            """Measure reprojection error for a physically valid candidate.

            Args:
                rvec: Candidate rotation vector.
                tvec: Candidate translation vector.
                coordinate_rms: OpenCV's per-coordinate RMS for this candidate, which
                    equals sqrt(mean squared error / 2). None projects the points.

            Returns:
                Mean squared pixel error, or infinity for an invalid candidate, and
                the squared per-corner distances when the points were projected.
            """
            # Plain-float checks: numpy's per-call overhead dominates on six values.
            if not all(
                map(math.isfinite, rvec.ravel().tolist() + tvec.ravel().tolist())
            ):
                return float("inf"), None
            rotation = cv2.Rodrigues(rvec)[0]
            # Only camera-space depth matters: every corner must be in front.
            if (object_points @ rotation[2] + tvec.flat[2]).min() <= 0:
                return float("inf"), None
            if coordinate_rms is not None:
                return 2.0 * float(coordinate_rms) ** 2, None
            projected = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                self.camera_matrix,
                self.distortion_coefficients,
            )[0]
            squared_distances = ((projected.reshape(-1, 2) - image_points) ** 2).sum(
                axis=1
            )
            return float(squared_distances.mean()), squared_distances

        if tag_count == 1 and previous_pose is not None:
            valid = [
                (rotation, translation, score(rotation, translation, rms)[0])
                for rotation, translation, rms in zip(
                    rotations, translations, initial_rms
                )
            ]
            valid = [candidate for candidate in valid if math.isfinite(candidate[2])]
            if not valid:
                return None
            # A subpixel fit tie is not enough on its own: both the initializer and
            # refined result must stay within the bounded neighborhood of the prior.
            best_rms = math.sqrt(min(candidate[2] for candidate in valid))
            tied = [
                candidate
                for candidate in valid
                if math.sqrt(candidate[2]) <= best_rms + 0.25
            ]
            previous_rotation = previous_pose[:3, :3]
            previous_translation = previous_pose[:3, 3]

            def pose_for(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
                """Undo object centering before comparing field-space camera poses."""
                rotation = cv2.Rodrigues(rvec)[0]
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = rotation.T
                pose[:3, 3] = object_origin - rotation.T @ tvec.reshape(3)
                return pose

            def separation(pose: np.ndarray) -> tuple[float, float]:
                """Return field translation and SO(3) rotation from the previous pose."""
                distance = np.linalg.norm(pose[:3, 3] - previous_translation)
                cosine = (np.trace(pose[:3, :3].T @ previous_rotation) - 1) / 2
                return distance, math.acos(min(max(cosine, -1.0), 1.0))

            def is_near_previous(distance: float, angle: float) -> bool:
                """Return whether a candidate remains inside continuity bounds."""
                return distance <= 1.0 and angle <= 0.35

            # Rank bounded candidates by one-meter translation and half-radian
            # rotation scales.
            bounded = []
            for rotation, translation, error in tied:
                distance, angle = separation(pose_for(rotation, translation))
                if is_near_previous(distance, angle):
                    continuity = float(distance**2 + (angle / 0.5) ** 2)
                    bounded.append((continuity, rotation, translation, error))
            if not bounded:
                return self._solve(geometry, image_points)
            _, rotation, translation, error = min(bounded, key=lambda c: c[0])
            distances = None
            if iterations:
                try:
                    refined_rotation, refined_translation = self._refine(
                        object_points, image_points, rotation, translation
                    )
                    refined_error, refined_distances = score(
                        refined_rotation, refined_translation
                    )
                    if refined_error <= error:
                        rotation, translation = refined_rotation, refined_translation
                        distances = refined_distances
                except cv2.error:
                    pass
            # The unrefined pick already passed the bounds check.
            if distances is not None and not is_near_previous(
                *separation(pose_for(rotation, translation))
            ):
                return self._solve(geometry, image_points)
            return rotation, translation, distances

        best = None
        best_error = float("inf")
        for rotation, translation, rms in zip(rotations, translations, initial_rms):
            error, distances = score(rotation, translation, rms)
            if not math.isfinite(error):
                continue
            if iterations:
                try:
                    refined_rotation, refined_translation = self._refine(
                        object_points, image_points, rotation, translation
                    )
                    refined_error, refined_distances = score(
                        refined_rotation, refined_translation
                    )
                    if refined_error <= error:
                        rotation, translation, error, distances = (
                            refined_rotation,
                            refined_translation,
                            refined_error,
                            refined_distances,
                        )
                except cv2.error:
                    pass  # A valid initializer is preferable to dropping this frame.
            if error < best_error:
                best = rotation, translation, distances
                best_error = error
        return best

    def _solution_quality(
        self,
        object_points: np.ndarray,
        tag_centers: np.ndarray,
        image_points: np.ndarray,
        rotation_vector: np.ndarray,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
        tag_count: int,
        squared_distances: np.ndarray | None = None,
    ) -> List[float]:
        """Measure how much a solved pose can be trusted.

        The robot-side pose estimator needs standard deviations, and these three
        numbers are what it derives them from. The solver's own scoring
        projection is reused when available; otherwise this costs one
        ``projectPoints`` call over at most a few dozen points.

        Args:
            object_points: Stacked field-space tag corners, shape (4 * tags, 3).
            tag_centers: Field-space center of each tag, shape (tags, 3).
            image_points: Matching image-space corners, shape (4 * tags, 2).
            rotation_vector: Rodrigues rotation from ``solvePnP``.
            rotation_matrix: Same rotation as a 3x3 matrix.
            translation_vector: Translation from ``solvePnP``.
            tag_count: Number of mapped tags that contributed to the solve.
            squared_distances: Squared per-corner reprojection distances already
                measured for this pose, or None to project the points here.

        Returns:
            ``[tag_count, mean_tag_distance_m, mean_reprojection_error_px]``.
        """
        if squared_distances is None:
            reprojected, _ = cv2.projectPoints(
                object_points,
                rotation_vector,
                translation_vector,
                self.camera_matrix,
                self.distortion_coefficients,
            )
            residuals = reprojected.reshape(-1, 2) - image_points
            squared_distances = (residuals * residuals).sum(axis=1)
        reprojection_error = float(np.sqrt(squared_distances).mean())

        camera_space_centers = (
            tag_centers @ rotation_matrix.T + translation_vector.reshape(3)
        )
        mean_distance = float(
            np.sqrt((camera_space_centers * camera_space_centers).sum(axis=1)).mean()
        )

        return [float(tag_count), mean_distance, reprojection_error]

    def estimate_pose_from_detections(
        self,
        detections: List[Detection],
        previous_pose: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, List[float]]]:
        """Estimate camera pose from AprilTag detections.

        Args:
            detections: List of AprilTag detections.
            previous_pose: Optional recent field-from-camera pose, not a smoothing target.

        Returns:
            The 4x4 camera pose in field coordinates paired with its quality
            metrics ``[tag_count, mean_tag_distance_m, reprojection_error_px]``,
            or None if pose estimation failed.
        """
        image_points_list = []
        tag_ids = []

        for detection in detections:
            tag_id = detection.tag_id
            # Unusable map entries are absent from _field_corners, so they are skipped.
            if tag_id not in self._field_corners or tag_id in tag_ids:
                continue

            corners = detection.corners.astype(np.float64, copy=False)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                continue

            tag_ids.append(tag_id)
            image_points_list.append(corners)

        if not tag_ids:
            return None

        image_points = np.concatenate(image_points_list)
        # Optimize around nearby tag geometry rather than the distant field origin.
        # This improves rotation/translation conditioning, particularly in older
        # OpenCV LM builds, without changing the published field coordinates.
        geometry = self._tag_set_geometry(tuple(tag_ids))
        solution = self._solve(geometry, image_points, previous_pose=previous_pose)
        if solution is None:
            return None
        rotation_vector, translation_vector, squared_distances = solution
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        translation_vector = translation_vector.reshape(3, 1) - (
            rotation_matrix @ geometry.origin
        ).reshape(3, 1)
        # Field-from-camera pose: the SE(3) inverse of camera-from-field, built
        # directly with the same arithmetic as fast_se3_inverse.
        global_camera_transform = np.eye(4, dtype=np.float64)
        global_camera_transform[:3, :3] = rotation_matrix.T
        global_camera_transform[:3, 3] = (
            -rotation_matrix.T @ translation_vector.reshape(3)
        )
        if not np.isfinite(global_camera_transform).all():
            return None

        quality = self._solution_quality(
            geometry.field_points,
            geometry.tag_centers,
            image_points,
            rotation_vector,
            rotation_matrix,
            translation_vector,
            geometry.tag_count,
            squared_distances,
        )
        return global_camera_transform, quality
