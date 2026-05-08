"""Oracle-temporal ROI detector with OpenCV-only tag verification (benchmark prototype).

This module implements a deliberately non-deployed benchmark path:

- **Temporal pose input**: Each frame uses only the camera pose estimated from
  previous-frame detections. Sequence starts and failed pose updates fall back to
  full-frame pupil detection.

- **No true AprilTag ID decode**: Expected ``tag_family`` / ``tag_id`` come from
  the known layout (``all_tags``). The verifier only checks that the image
  region looks like a tag before emitting a detection.

- **Visual verification**: Detections are returned only when at least one enabled
  verification path accepts the ROI (warped border/contrast, contour quad, or a
  lightweight gate for sub-pixel-only mode).

Supported dataset assumptions: ``tag36h11`` family and the same intrinsics /
``tag_size_m`` conventions as ``PupilAprilTagDetector`` / ``TemporalPupilAprilTagDetector``.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from ..utils import GroundTruthTag, blender_world_to_cv_camera
from .base import CameraIntrinsics, TagDetection
from .pupil_detector import PupilAprilTagDetector


def _world_points_to_image_pixels(
    points_world_m: np.ndarray,
    camera_matrix_world_blender: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Projects world points and returns pixels plus OpenCV camera-space depth."""
    blender_to_cv_local = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    world_from_cv_camera = np.asarray(camera_matrix_world_blender, dtype=np.float64) @ blender_to_cv_local
    camera_from_world = np.linalg.inv(world_from_cv_camera)
    points_world_h = np.concatenate(
        [points_world_m.reshape(-1, 3).astype(np.float64), np.ones((points_world_m.reshape(-1, 3).shape[0], 1))],
        axis=1,
    )
    points_camera = (camera_from_world @ points_world_h.T).T[:, :3]
    z_camera = points_camera[:, 2].copy()
    camera_matrix = np.asarray(
        [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pixels = np.empty((points_camera.shape[0], 2), dtype=np.float64)
    pixels[:, 0] = intrinsics.fx * (points_camera[:, 0] / points_camera[:, 2]) + intrinsics.cx
    pixels[:, 1] = intrinsics.fy * (points_camera[:, 1] / points_camera[:, 2]) + intrinsics.cy
    return pixels, z_camera


def _quad_area_pixels(corners_xy: np.ndarray) -> float:
    """Returns the absolute polygon area of a quad in pixel space."""
    contour = corners_xy.reshape(-1, 1, 2).astype(np.float32)
    return float(abs(cv2.contourArea(contour)))


def _axis_aligned_roi_from_quad(
    corners_xy: np.ndarray,
    image_width: int,
    image_height: int,
    padding_factor: float,
    min_side_px: int,
) -> tuple[int, int, int, int]:
    """Returns ``(x, y, w, h)`` ROI enclosing the quad with padding and size floors."""
    min_xy = np.min(corners_xy, axis=0)
    max_xy = np.max(corners_xy, axis=0)
    center_xy = 0.5 * (min_xy + max_xy)
    span = max_xy - min_xy
    max_span = float(np.max(span))
    pad = max_span * max(0.0, float(padding_factor) - 1.0) * 0.5
    half_w = max(min_side_px * 0.5, span[0] * 0.5 + pad)
    half_h = max(min_side_px * 0.5, span[1] * 0.5 + pad)
    x1 = int(np.floor(center_xy[0] - half_w))
    y1 = int(np.floor(center_xy[1] - half_h))
    x2 = int(np.ceil(center_xy[0] + half_w))
    y2 = int(np.ceil(center_xy[1] + half_h))
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(x1 + 1, min(image_width, x2))
    y2 = max(y1 + 1, min(image_height, y2))
    return (x1, y1, x2 - x1, y2 - y1)


def _merge_axis_aligned_regions(
    regions: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merges overlapping axis-aligned ROIs into larger boxes."""
    if not regions:
        return []
    boxes = [[region[0], region[1], region[0] + region[2], region[1] + region[3]] for region in regions]
    changed = True
    while changed:
        changed = False
        merged: list[list[int]] = []
        for box in boxes:
            for existing in merged:
                horizontal_gap = box[0] > existing[2] or box[2] < existing[0]
                vertical_gap = box[1] > existing[3] or box[3] < existing[1]
                if not (horizontal_gap or vertical_gap):
                    existing[0] = min(existing[0], box[0])
                    existing[1] = min(existing[1], box[1])
                    existing[2] = max(existing[2], box[2])
                    existing[3] = max(existing[3], box[3])
                    changed = True
                    break
            else:
                merged.append(box)
        boxes = merged
    return [(int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])) for box in boxes]


def _sort_quad_clockwise_xy(points_xy: np.ndarray) -> np.ndarray:
    """Sorts four 2D points clockwise starting from the smallest polar angle."""
    center = np.mean(points_xy, axis=0)
    angles = np.arctan2(points_xy[:, 1] - center[1], points_xy[:, 0] - center[0])
    order = np.argsort(angles)
    return points_xy[order].astype(np.float64)


def _best_cyclic_alignment_error(reference_xy: np.ndarray, candidate_xy: np.ndarray) -> tuple[float, int]:
    """Returns minimum mean corner distance over cyclic permutations of ``candidate_xy``."""
    reference_sorted = _sort_quad_clockwise_xy(reference_xy)
    candidate_sorted = _sort_quad_clockwise_xy(candidate_xy)
    best_error = float("inf")
    best_shift = 0
    for shift in range(4):
        rolled = np.roll(candidate_sorted, shift, axis=0)
        mean_distance = float(np.mean(np.linalg.norm(rolled - reference_sorted, axis=1)))
        if mean_distance < best_error:
            best_error = mean_distance
            best_shift = shift
    return best_error, best_shift


def _align_quad_to_reference(reference_xy: np.ndarray, candidate_xy: np.ndarray) -> np.ndarray | None:
    """Reorders ``candidate_xy`` corners to best match ``reference_xy``."""
    error, shift = _best_cyclic_alignment_error(reference_xy, candidate_xy)
    max_reference_span = float(np.max(np.linalg.norm(reference_xy - np.mean(reference_xy, axis=0), axis=1)))
    if error > 0.45 * max(32.0, max_reference_span):
        return None
    candidate_sorted = _sort_quad_clockwise_xy(candidate_xy)
    return np.roll(candidate_sorted, shift, axis=0)


def verify_warp_border_contrast(
    gray_roi_u8: np.ndarray,
    quad_xy_local: np.ndarray,
    canonical_size: int,
    min_border_delta: float,
    min_inner_std: float,
) -> bool:
    """Warps the predicted quad to a square and checks border brightness vs interior texture."""
    if gray_roi_u8.size == 0 or quad_xy_local.shape != (4, 2):
        return False
    destination = np.asarray(
        [
            [0.0, 0.0],
            [canonical_size - 1.0, 0.0],
            [canonical_size - 1.0, canonical_size - 1.0],
            [0.0, canonical_size - 1.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(quad_xy_local.astype(np.float32), destination)
    warped = cv2.warpPerspective(gray_roi_u8, transform, (canonical_size, canonical_size))
    border_mask = np.zeros((canonical_size, canonical_size), dtype=np.uint8)
    border_thickness = max(2, canonical_size // 16)
    border_mask[:border_thickness, :] = 1
    border_mask[-border_thickness:, :] = 1
    border_mask[:, :border_thickness] = 1
    border_mask[:, -border_thickness:] = 1
    inner_mask = 1 - border_mask
    border_values = warped[border_mask.astype(bool)]
    inner_values = warped[inner_mask.astype(bool)]
    if border_values.size < 8 or inner_values.size < 8:
        return False
    border_mean = float(np.mean(border_values))
    inner_mean = float(np.mean(inner_values))
    inner_std = float(np.std(inner_values))
    border_std = float(np.std(border_values))
    # The original prototype assumed a bright outer margin. Real AprilTag crops
    # can contain either the white margin or the black payload border depending
    # on which quad is being tracked, so accept either contrast polarity while
    # still requiring a textured interior. A low-variance border is an extra
    # AprilTag-like cue, but keep it permissive for motion-blurred frames.
    mean_delta = abs(border_mean - inner_mean)
    border_is_coherent = border_std < max(45.0, inner_std * 1.8)
    has_tag_like_contrast = mean_delta > min_border_delta or inner_std > min_inner_std * 1.35
    return has_tag_like_contrast and inner_std > min_inner_std * 0.55 and border_is_coherent


def refine_corners_subpix(
    gray_u8: np.ndarray,
    corners_xy_image: np.ndarray,
    window_half_size: int,
    max_iterations: int,
    epsilon: float,
) -> np.ndarray:
    """Refines corner locations with ``cv2.cornerSubPix`` in the full image."""
    height, width = gray_u8.shape[:2]
    refined = corners_xy_image.reshape(1, 4, 2).astype(np.float32).copy()
    half_w = max(1, int(window_half_size))
    half_h = max(1, int(window_half_size))
    win_w = max(3, half_w * 2 + 1)
    win_h = max(3, half_h * 2 + 1)
    u_lo = float(win_w // 2)
    u_hi = float(width - 1 - win_w // 2)
    v_lo = float(win_h // 2)
    v_hi = float(height - 1 - win_h // 2)
    if u_hi <= u_lo or v_hi <= v_lo:
        return corners_xy_image.astype(np.float64)
    refined[0, :, 0] = np.clip(refined[0, :, 0], u_lo, u_hi)
    refined[0, :, 1] = np.clip(refined[0, :, 1], v_lo, v_hi)
    win = (win_w, win_h)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(max_iterations), float(epsilon))
    cv2.cornerSubPix(gray_u8, refined, win, (-1, -1), criteria)
    return refined.reshape(4, 2).astype(np.float64)


def verify_contour_quad(
    gray_roi_u8: np.ndarray,
    predicted_quad_local: np.ndarray,
    max_mean_reprojection_px: float,
) -> np.ndarray | None:
    """Fits a quadrilateral from Canny edges; returns aligned corners in ROI coordinates or ``None``."""
    blurred = cv2.GaussianBlur(gray_roi_u8, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_quad: np.ndarray | None = None
    best_area = 0.0
    height_roi, width_roi = gray_roi_u8.shape[:2]
    min_area = max(64.0, 0.02 * float(height_roi * width_roi))
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 12.0:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4:
            continue
        area = abs(cv2.contourArea(approx))
        if area < min_area or area <= best_area:
            continue
        candidate = approx.reshape(4, 2).astype(np.float64)
        aligned = _align_quad_to_reference(predicted_quad_local, candidate)
        if aligned is None:
            continue
        mean_distance = float(np.mean(np.linalg.norm(aligned - predicted_quad_local, axis=1)))
        if mean_distance > max_mean_reprojection_px:
            continue
        best_area = area
        best_quad = aligned
    return best_quad


def verify_local_contrast_gate(
    gray_u8: np.ndarray,
    corners_xy_image: np.ndarray,
    patch_radius_px: int,
    min_range: float,
) -> bool:
    """Requires each corner neighborhood to have sufficient intensity range."""
    height, width = gray_u8.shape[:2]
    for corner in corners_xy_image:
        u = int(round(float(corner[0])))
        v = int(round(float(corner[1])))
        y1 = max(0, v - patch_radius_px)
        y2 = min(height, v + patch_radius_px + 1)
        x1 = max(0, u - patch_radius_px)
        x2 = min(width, u + patch_radius_px + 1)
        patch = gray_u8[y1:y2, x1:x2]
        if patch.size == 0:
            return False
        if float(patch.max()) - float(patch.min()) < min_range:
            return False
    return True


class FastTemporalCustomAprilTagDetector:
    """Temporal-ROI tag tracker using oracle pose and OpenCV verification (benchmark-only)."""

    name = "fast-temporal-custom-apriltags"

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 4,
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.8,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
        padding_factor: float = 2.0,
        max_regions: int = 24,
        min_region_size_px: int = 28,
        merge_overlapping_rois: bool = True,
        min_detection_count: int | None = 1,
        verify_modes: str = "corner_subpix,warp_contrast",
        warp_canonical_size: int = 48,
        warp_min_border_delta: float = 6.0,
        warp_min_inner_std: float = 7.0,
        subpix_window_half_size: int = 5,
        subpix_max_iterations: int = 40,
        subpix_epsilon: float = 0.01,
        subpix_max_shift_px: float = 4.0,
        contour_max_mean_error_px: float = 18.0,
        contrast_gate_min_range: float = 18.0,
        contrast_gate_patch_radius_px: int = 7,
        enable_photometric_refine: bool = False,
        pose_source: str = "solvepnp_corners",
        optical_tracking_mode: str = "tag_plane",
        keyframe_interval: int = 10,
    ) -> None:
        self.families = str(families)
        self.padding_factor = float(padding_factor)
        self.max_regions = int(max_regions)
        self.min_region_size_px = int(min_region_size_px)
        self.merge_overlapping_rois = bool(merge_overlapping_rois)
        self.min_detection_count = min_detection_count
        self.verify_modes = [part.strip() for part in verify_modes.split(",") if part.strip()]
        self.warp_canonical_size = int(warp_canonical_size)
        self.warp_min_border_delta = float(warp_min_border_delta)
        self.warp_min_inner_std = float(warp_min_inner_std)
        self.subpix_window_half_size = int(subpix_window_half_size)
        self.subpix_max_iterations = int(subpix_max_iterations)
        self.subpix_epsilon = float(subpix_epsilon)
        self.subpix_max_shift_px = float(subpix_max_shift_px)
        self.contour_max_mean_error_px = float(contour_max_mean_error_px)
        self.contrast_gate_min_range = float(contrast_gate_min_range)
        self.contrast_gate_patch_radius_px = int(contrast_gate_patch_radius_px)
        self.enable_photometric_refine = bool(enable_photometric_refine)
        if pose_source not in ("oracle_center_world", "solvepnp_corners"):
            raise ValueError("pose_source must be oracle_center_world or solvepnp_corners")
        self.pose_source = str(pose_source)
        if optical_tracking_mode not in ("tag_plane", "corners", "none"):
            raise ValueError("optical_tracking_mode must be tag_plane, corners, or none")
        self.optical_tracking_mode = str(optical_tracking_mode)
        self.keyframe_interval = max(0, int(keyframe_interval))
        self._frames_since_keyframe = 0
        self._full_pupil = PupilAprilTagDetector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
        )
        self._intrinsics: CameraIntrinsics | None = None
        self._camera_matrix_world: np.ndarray | None = None
        self._previous_world_from_cv_camera: np.ndarray | None = None
        self._current_world_from_cv_camera: np.ndarray | None = None
        self._has_pose_for_next_frame = False
        self._layout_tags: list[GroundTruthTag] = []
        self._last_sequence: str | None = None
        self._sequence_requires_full_pupil: bool = True
        self._coverage_values: list[float] = []
        self._region_counts: list[int] = []
        self._pupil_full_frame_calls = 0
        self._fallback_frames = 0
        self._custom_accept_frames = 0
        self._verified_tag_count_total = 0
        self._photometric_iterations_total = 0
        self._previous_gray: np.ndarray | None = None
        self._previous_tag_corners: dict[tuple[str, int], np.ndarray] = {}
        self._lk_grid_side = 7
        self.last_regions: list[tuple[int, int, int, int]] = []
        self.last_raw_regions: list[tuple[int, int, int, int]] = []
        self.last_projected_tags: list[tuple[str, int, np.ndarray]] = []

    def prepare_frame(
        self,
        intrinsics: CameraIntrinsics,
        all_tags: Sequence[GroundTruthTag],
        camera_matrix_world: np.ndarray | None,
        *,
        sequence: str | None = None,
    ) -> None:
        """Caches intrinsics/layout; temporal pose is supplied by previous detections."""
        if sequence is not None and sequence != self._last_sequence:
            self._last_sequence = sequence
            self._sequence_requires_full_pupil = True
            self._has_pose_for_next_frame = False
            self._camera_matrix_world = None
            self._previous_world_from_cv_camera = None
            self._current_world_from_cv_camera = None
            self._clear_optical_flow_tracks()
            self._frames_since_keyframe = 0
        self._intrinsics = intrinsics
        self._layout_tags = list(all_tags)

    def update_pose_from_detections(
        self,
        world_from_cv_camera: np.ndarray | None,
        *,
        sequence: str | None = None,
    ) -> None:
        """Stores a predicted next-frame camera pose for temporal ROIs."""
        if sequence is not None and sequence != self._last_sequence:
            self._last_sequence = sequence
            self._sequence_requires_full_pupil = True
            self._has_pose_for_next_frame = False
            self._camera_matrix_world = None
            self._previous_world_from_cv_camera = None
            self._current_world_from_cv_camera = None
            self._clear_optical_flow_tracks()
            self._frames_since_keyframe = 0
            return
        if world_from_cv_camera is None:
            self._has_pose_for_next_frame = False
            self._camera_matrix_world = None
            self._previous_world_from_cv_camera = None
            self._current_world_from_cv_camera = None
            self._sequence_requires_full_pupil = True
            self._clear_optical_flow_tracks()
            self._frames_since_keyframe = 0
            return

        world_from_cv_camera = np.asarray(world_from_cv_camera, dtype=np.float64).reshape(4, 4)
        self._previous_world_from_cv_camera = self._current_world_from_cv_camera
        self._current_world_from_cv_camera = world_from_cv_camera

        predicted_world_from_cv_camera = world_from_cv_camera
        if self._previous_world_from_cv_camera is not None:
            # Constant-velocity pose prediction. Projecting frame N+1 with frame
            # N's pose causes a global offset when the camera moves.
            world_delta = world_from_cv_camera @ np.linalg.inv(self._previous_world_from_cv_camera)
            predicted_world_from_cv_camera = world_delta @ world_from_cv_camera

        cv_to_blender_local = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
        self._camera_matrix_world = predicted_world_from_cv_camera @ cv_to_blender_local
        self._has_pose_for_next_frame = True
        self._sequence_requires_full_pupil = False
        if hasattr(self, "_rust"):
            self._rust.set_pose_blender_row_major(self._camera_matrix_world.reshape(16).tolist())

    def close(self) -> None:
        """Releases native pupil resources used for full-frame fallback."""
        self._full_pupil.close()

    def acceleration_report(self) -> dict[str, float | int | str]:
        """Returns ROI, fallback, and verification diagnostics for benchmark JSON."""
        mean_coverage = float(np.mean(self._coverage_values)) if self._coverage_values else 1.0
        mean_regions = float(np.mean(self._region_counts)) if self._region_counts else 0.0
        frames_with_custom = int(self._custom_accept_frames)
        mean_verified_per_custom_frame = (
            float(self._verified_tag_count_total) / float(frames_with_custom) if frames_with_custom > 0 else 0.0
        )
        return {
            "mean_roi_coverage": mean_coverage,
            "min_roi_coverage": float(np.min(self._coverage_values)) if self._coverage_values else 1.0,
            "max_roi_coverage": float(np.max(self._coverage_values)) if self._coverage_values else 1.0,
            "mean_regions": mean_regions,
            "pupil_full_frame_calls": int(self._pupil_full_frame_calls),
            "fallback_frames": int(self._fallback_frames),
            "custom_primary_frames": int(self._custom_accept_frames),
            "mean_verified_tags_per_custom_frame": mean_verified_per_custom_frame,
            "verify_modes": ",".join(self.verify_modes),
            "oracle_pose_upper_bound": 0,
            "no_bitwise_id_decode": 1,
            "photometric_refine_enabled": int(self.enable_photometric_refine),
            "photometric_extra_iterations_total": int(self._photometric_iterations_total),
            "pose_source": self.pose_source,
            "optical_tracking_mode": str(self.optical_tracking_mode),
            "keyframe_interval": int(self.keyframe_interval),
        }

    def _clear_optical_flow_tracks(self) -> None:
        """Drops image-space temporal tracks when a sequence/pose is reset."""
        self._previous_gray = None
        self._previous_tag_corners = {}

    def _remember_optical_flow_tracks(
        self,
        gray_u8: np.ndarray,
        detections: Sequence[TagDetection],
    ) -> None:
        """Stores accepted/full detections as anchors for next-frame planar flow."""
        self._previous_gray = gray_u8.copy()
        self._previous_tag_corners = {
            (str(det.tag_family), int(det.tag_id)): np.asarray(det.corners, dtype=np.float64)
            .reshape(4, 2)
            .copy()
            for det in detections
        }

    def _sample_quad_plane_points(self, corners_xy: np.ndarray) -> np.ndarray:
        """Returns a stable grid of points on the previous tag plane for LK tracking."""
        grid_side = max(3, int(self._lk_grid_side))
        canonical = []
        for y in np.linspace(0.0, 1.0, grid_side):
            for x in np.linspace(0.0, 1.0, grid_side):
                canonical.append([x, y])
        canonical_xy = np.asarray(canonical, dtype=np.float32).reshape(-1, 1, 2)
        square = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(square, corners_xy.astype(np.float32))
        return cv2.perspectiveTransform(canonical_xy, transform).reshape(-1, 2).astype(np.float32)

    def _track_corners_with_optical_flow(
        self,
        current_gray_u8: np.ndarray,
        previous_corners_xy: np.ndarray,
    ) -> np.ndarray | None:
        """Tracks only the four tag corners with LK optical flow."""
        previous_gray = self._previous_gray
        if previous_gray is None or previous_gray.shape[:2] != current_gray_u8.shape[:2]:
            return None
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray_u8,
            previous_corners_xy.astype(np.float32).reshape(-1, 1, 2),
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            minEigThreshold=1e-4,
        )
        if next_points is None or status is None or int(np.count_nonzero(status)) < 4:
            return None
        if errors is not None and np.any(errors.reshape(-1) > 90.0):
            return None
        projected = next_points.reshape(4, 2).astype(np.float64)
        if not np.all(np.isfinite(projected)):
            return None
        old_area = _quad_area_pixels(previous_corners_xy)
        new_area = _quad_area_pixels(projected)
        if (
            old_area < 4.0
            or new_area < 4.0
            or new_area > old_area * 4.0
            or new_area < old_area * 0.20
        ):
            return None
        height, width = current_gray_u8.shape[:2]
        max_motion = 0.60 * float(max(width, height))
        if float(np.max(np.linalg.norm(projected - previous_corners_xy, axis=1))) > max_motion:
            return None
        return projected

    def _track_plane_with_optical_flow(
        self,
        current_gray_u8: np.ndarray,
        previous_corners_xy: np.ndarray,
    ) -> np.ndarray | None:
        """Tracks many points on a tag plane and projects the previous corners by a RANSAC homography."""
        previous_gray = self._previous_gray
        if previous_gray is None or previous_gray.shape[:2] != current_gray_u8.shape[:2]:
            return None
        prev_points = self._sample_quad_plane_points(previous_corners_xy)
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray_u8,
            prev_points.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=0,
            minEigThreshold=1e-4,
        )
        if next_points is None or status is None:
            return None
        status_mask = status.reshape(-1).astype(bool)
        if errors is not None:
            # Keep this permissive: LK's error scale varies with blur/exposure,
            # and RANSAC below removes geometrically inconsistent points.
            status_mask &= errors.reshape(-1) < 90.0
        height, width = current_gray_u8.shape[:2]
        tracked = next_points.reshape(-1, 2)
        status_mask &= (tracked[:, 0] >= -4.0) & (tracked[:, 0] < width + 4.0)
        status_mask &= (tracked[:, 1] >= -4.0) & (tracked[:, 1] < height + 4.0)
        if int(np.count_nonzero(status_mask)) < 8:
            return None
        homography, inliers = cv2.findHomography(
            prev_points[status_mask].astype(np.float32),
            tracked[status_mask].astype(np.float32),
            cv2.RANSAC,
            5.0,
        )
        if homography is None or inliers is None:
            return None
        inlier_ratio = float(np.count_nonzero(inliers)) / float(inliers.size)
        if inlier_ratio < 0.40:
            return None
        projected = cv2.perspectiveTransform(
            previous_corners_xy.astype(np.float32).reshape(-1, 1, 2),
            homography,
        ).reshape(4, 2).astype(np.float64)
        if not np.all(np.isfinite(projected)):
            return None
        old_area = _quad_area_pixels(previous_corners_xy)
        new_area = _quad_area_pixels(projected)
        if (
            old_area < 4.0
            or new_area < 4.0
            or new_area > old_area * 4.0
            or new_area < old_area * 0.20
        ):
            return None
        max_motion = 0.60 * float(max(width, height))
        if float(np.max(np.linalg.norm(projected - previous_corners_xy, axis=1))) > max_motion:
            return None
        return projected

    def _flow_projected_layout(
        self,
        gray_u8: np.ndarray,
        pose_projected: Sequence[tuple[str, int, np.ndarray]],
    ) -> list[tuple[str, int, np.ndarray]]:
        """Uses tag-plane optical flow for known previous tags, with pose projections as fallback."""
        pose_by_key = {(str(family), int(tag_id)): corners for family, tag_id, corners in pose_projected}
        tracked: list[tuple[str, int, np.ndarray]] = []
        used: set[tuple[str, int]] = set()
        for key, previous_corners in self._previous_tag_corners.items():
            mode = str(getattr(self, "optical_tracking_mode", "tag_plane"))
            if mode == "none":
                tracked_corners = None
            elif mode == "corners":
                tracked_corners = self._track_corners_with_optical_flow(gray_u8, previous_corners)
            else:
                tracked_corners = self._track_plane_with_optical_flow(gray_u8, previous_corners)
            if tracked_corners is None:
                continue
            if key in pose_by_key:
                pose_corners = pose_by_key[key]
                mean_delta = float(np.mean(np.linalg.norm(tracked_corners - pose_corners, axis=1)))
                max_span = float(np.max(np.linalg.norm(pose_corners - np.mean(pose_corners, axis=0), axis=1)))
                if mean_delta > max(45.0, 1.4 * max_span):
                    continue
            family, tag_id = key
            tracked.append((family, int(tag_id), tracked_corners))
            used.add(key)
        for family, tag_id, corners in pose_projected:
            key = (str(family), int(tag_id))
            if key not in used:
                tracked.append((str(family), int(tag_id), corners))
        return tracked[: self.max_regions]

    def detect(self, image_bgr: np.ndarray, intrinsics: CameraIntrinsics, tag_size_m: float) -> list[TagDetection]:
        """Runs full-image pupil on sequence start or fallback; otherwise custom ROI verification."""
        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        camera_matrix = np.asarray(
            [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        force_keyframe = self.keyframe_interval > 0 and self._frames_since_keyframe >= self.keyframe_interval
        if (
            force_keyframe
            or self._sequence_requires_full_pupil
            or self._intrinsics is None
            or self._camera_matrix_world is None
            or not self._has_pose_for_next_frame
        ):
            self._pupil_full_frame_calls += 1
            self._sequence_requires_full_pupil = False
            self._coverage_values.append(1.0)
            self._region_counts.append(1)
            self.last_regions = [(0, 0, width, height)]
            self.last_raw_regions = [(0, 0, width, height)]
            self.last_projected_tags = []
            detections = self._full_pupil.detect(image_bgr, intrinsics, tag_size_m)
            self._remember_optical_flow_tracks(gray, detections)
            self._frames_since_keyframe = 0
            return detections

        pose_projected = self._projected_layout(width, height)
        projected = self._flow_projected_layout(gray, pose_projected)
        self.last_projected_tags = [(family, int(tag_id), corners.copy()) for family, tag_id, corners in projected]
        per_tag_rois = [
            _axis_aligned_roi_from_quad(corners, width, height, self.padding_factor, self.min_region_size_px)
            for _family, _tag_id, corners in projected
        ]
        self.last_raw_regions = list(per_tag_rois)
        regions_for_coverage = (
            _merge_axis_aligned_regions(list(per_tag_rois)) if self.merge_overlapping_rois else list(per_tag_rois)
        )
        self.last_regions = list(regions_for_coverage)
        image_area = max(1, width * height)
        roi_area = sum(region[2] * region[3] for region in regions_for_coverage)
        self._coverage_values.append(min(1.0, roi_area / float(image_area)))
        self._region_counts.append(float(len(per_tag_rois)))

        custom_detections: list[TagDetection] = []
        dedup_keys: set[tuple[str, int]] = set()
        for (tag_family, tag_id, corners_image), (region_x, region_y, region_w, region_h) in zip(
            projected, per_tag_rois, strict=True
        ):
            dedup_key = (tag_family, int(tag_id))
            if dedup_key in dedup_keys:
                continue
            verified_corners = self._verify_tag_in_roi(
                gray,
                corners_image,
                region_x,
                region_y,
                region_w,
                region_h,
            )
            if verified_corners is None:
                continue
            dedup_keys.add(dedup_key)
            pose_t = self._pose_translation_m(
                tag_family,
                tag_id,
                verified_corners,
                camera_matrix,
                dist_coeffs,
                tag_size_m,
            )
            custom_detections.append(
                TagDetection(
                    tag_family=tag_family,
                    tag_id=int(tag_id),
                    corners=verified_corners,
                    center=np.mean(verified_corners, axis=0),
                    pose_t=pose_t,
                    decision_margin=None,
                    hamming=None,
                )
            )

        min_count = self.min_detection_count
        if min_count is not None and len(custom_detections) < int(min_count):
            self._pupil_full_frame_calls += 1
            self._fallback_frames += 1
            self._coverage_values[-1] = 1.0
            self._region_counts[-1] = 1
            self.last_regions = [(0, 0, width, height)]
            self.last_raw_regions = [(0, 0, width, height)]
            detections = self._full_pupil.detect(image_bgr, intrinsics, tag_size_m)
            self._remember_optical_flow_tracks(gray, detections)
            self._frames_since_keyframe = 0
            return detections

        self._custom_accept_frames += 1
        self._verified_tag_count_total += len(custom_detections)
        self._remember_optical_flow_tracks(gray, custom_detections)
        self._frames_since_keyframe += 1
        return custom_detections

    def _projected_layout(self, image_width: int, image_height: int) -> list[tuple[str, int, np.ndarray]]:
        """Projects all layout tags; keeps those with finite quads mostly inside the image."""
        intrinsics = self._intrinsics
        camera_matrix_world = self._camera_matrix_world
        if intrinsics is None or camera_matrix_world is None:
            return []
        best_by_key: dict[tuple[str, int], tuple[float, str, int, np.ndarray]] = {}
        for tag in self._layout_tags:
            if tag.corners_world is None:
                continue
            corners_world_full = np.asarray(tag.corners_world, dtype=np.float64).reshape(4, 3)
            if tag.center_world is not None:
                center_world = np.asarray(tag.center_world, dtype=np.float64).reshape(1, 3)
                corners_world = center_world + (corners_world_full - center_world) * 0.78
            else:
                corners_world = corners_world_full
            corners_image, z_camera = _world_points_to_image_pixels(corners_world, camera_matrix_world, intrinsics)
            if not (np.all(np.isfinite(corners_image)) and np.all(np.isfinite(z_camera))):
                continue
            # OpenCV camera coordinates look down +Z. Reject tags with any
            # projected payload corner behind/too close to the camera; otherwise
            # perspective division mirrors them into plausible-looking screen ROIs.
            if np.any(z_camera <= 1e-4):
                continue
            margin = 0.02 * float(max(image_width, image_height))
            inside = np.sum(
                (corners_image[:, 0] >= -margin)
                & (corners_image[:, 0] < image_width + margin)
                & (corners_image[:, 1] >= -margin)
                & (corners_image[:, 1] < image_height + margin)
            )
            if inside < 3:
                continue
            area = _quad_area_pixels(corners_image)
            if area < 4.0:
                continue
            key = (str(tag.tag_family), int(tag.tag_id))
            candidate = (area, str(tag.tag_family), int(tag.tag_id), corners_image.astype(np.float64))
            previous = best_by_key.get(key)
            if previous is None or area > previous[0]:
                best_by_key[key] = candidate
        scored = sorted(best_by_key.values(), key=lambda item: item[0], reverse=True)
        return [(family, tag_id, corners) for _area, family, tag_id, corners in scored[: self.max_regions]]

    def _verify_tag_in_roi(
        self,
        gray_u8: np.ndarray,
        corners_xy_image: np.ndarray,
        region_x: int,
        region_y: int,
        region_w: int,
        region_h: int,
    ) -> np.ndarray | None:
        """Applies enabled verification modes; returns image-space corners or ``None``."""
        crop = gray_u8[region_y : region_y + region_h, region_x : region_x + region_w]
        quad_local = corners_xy_image - np.asarray([region_x, region_y], dtype=np.float64)
        modes = self.verify_modes
        if not modes:
            return None
        warp_mode_index = modes.index("warp_contrast") if "warp_contrast" in modes else None
        subpix_mode_index = modes.index("corner_subpix") if "corner_subpix" in modes else None
        passed_warp = False
        working_local = quad_local.astype(np.float64)
        for mode in modes:
            if mode == "warp_contrast":
                passed_warp = verify_warp_border_contrast(
                    crop,
                    working_local,
                    self.warp_canonical_size,
                    self.warp_min_border_delta,
                    self.warp_min_inner_std,
                )
                if not passed_warp:
                    return None
            elif mode == "corner_subpix":
                corners_image = working_local + np.asarray([region_x, region_y], dtype=np.float64)
                requires_prior_warp = (
                    warp_mode_index is not None
                    and subpix_mode_index is not None
                    and warp_mode_index < subpix_mode_index
                )
                if requires_prior_warp and not passed_warp:
                    return None
                if not requires_prior_warp:
                    if not verify_local_contrast_gate(
                        gray_u8,
                        corners_image,
                        self.contrast_gate_patch_radius_px,
                        self.contrast_gate_min_range,
                    ):
                        return None
                refined = refine_corners_subpix(
                    gray_u8,
                    corners_image,
                    self.subpix_window_half_size,
                    self.subpix_max_iterations,
                    self.subpix_epsilon,
                )
                shift_px = np.linalg.norm(refined - corners_image, axis=1)
                if np.any(shift_px > self.subpix_max_shift_px):
                    return None
                working_local = refined - np.asarray([region_x, region_y], dtype=np.float64)
            elif mode == "contour_quad":
                fitted = verify_contour_quad(crop, working_local, self.contour_max_mean_error_px)
                if fitted is None:
                    return None
                working_local = fitted
            elif mode == "contrast_gate":
                corners_image = working_local + np.asarray([region_x, region_y], dtype=np.float64)
                if not verify_local_contrast_gate(
                    gray_u8,
                    corners_image,
                    self.contrast_gate_patch_radius_px,
                    self.contrast_gate_min_range,
                ):
                    return None
            elif mode == "photometric_refine":
                if not self.enable_photometric_refine:
                    continue
                corners_image = working_local + np.asarray([region_x, region_y], dtype=np.float64)
                refined = refine_corners_subpix(
                    gray_u8,
                    corners_image,
                    self.subpix_window_half_size + 1,
                    max(10, self.subpix_max_iterations // 2),
                    self.subpix_epsilon * 0.5,
                )
                shift_px = np.linalg.norm(refined - corners_image, axis=1)
                if np.any(shift_px > self.subpix_max_shift_px):
                    return None
                working_local = refined - np.asarray([region_x, region_y], dtype=np.float64)
                self._photometric_iterations_total += 1
        return working_local + np.asarray([region_x, region_y], dtype=np.float64)

    def _translation_from_oracle_center_world(self, tag_family: str, tag_id: int) -> np.ndarray | None:
        """Returns tag center in OpenCV camera coordinates using Blender metadata (oracle)."""
        camera_matrix_world = self._camera_matrix_world
        if camera_matrix_world is None:
            return None
        matrix_rows = camera_matrix_world.astype(np.float64).tolist()
        for tag in self._layout_tags:
            if str(tag.tag_family) != str(tag_family) or int(tag.tag_id) != int(tag_id):
                continue
            if tag.center_world is None:
                return None
            center_world_list = np.asarray(tag.center_world, dtype=float).reshape(3).tolist()
            translation = blender_world_to_cv_camera(matrix_rows, center_world_list)
            return np.asarray(translation, dtype=np.float64).reshape(3)
        return None

    def _pose_translation_m(
        self,
        tag_family: str,
        tag_id: int,
        refined_corners: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tag_size_m: float,
    ) -> np.ndarray | None:
        """Pose translation aligned with benchmark GT (oracle center) or from ``solvePnP``."""
        if self.pose_source == "oracle_center_world":
            oracle_translation = self._translation_from_oracle_center_world(tag_family, tag_id)
            if oracle_translation is not None:
                return oracle_translation
        return self._estimate_pose_t_from_corners(refined_corners, camera_matrix, dist_coeffs, tag_size_m)

    def _estimate_pose_t_from_corners(
        self,
        corners: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tag_size_m: float,
    ) -> np.ndarray | None:
        """Estimates tag translation from refined corners (same model as temporal pupil)."""
        payload_edge_m = float(tag_size_m) * 0.78
        half = payload_edge_m * 0.5
        object_points = np.asarray(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float64,
        )
        ok, _rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            corners.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        return None if not ok else translation_vector.reshape(3)
