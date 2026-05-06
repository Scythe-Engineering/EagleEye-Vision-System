"""Oracle-temporal ROI detector with OpenCV-only tag verification (benchmark prototype).

This module implements a deliberately non-deployed benchmark path:

- **Oracle pose (upper bound)**: Each frame uses ``prepare_frame`` with synthetic
  metadata ``camera_matrix_world`` so projected tag quads match ground truth
  camera motion. This measures how fast a temporal-ROI + custom verifier stack
  could run if pose were perfect, not a realistic on-robot tracker.

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
) -> np.ndarray:
    """Projects 3D world points into pixel coordinates using the Blender camera matrix.

    Args:
        points_world_m: Array of shape ``(n, 3)`` with world-space points in meters.
        camera_matrix_world_blender: ``4x4`` Blender ``world`` from ``camera`` matrix.
        intrinsics: Pinhole intrinsics for the rendered image.

    Returns:
        Array of shape ``(n, 2)`` with floating-point pixel coordinates ``(u, v)``.
    """
    blender_to_cv_local = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    world_from_cv_camera = np.asarray(camera_matrix_world_blender, dtype=np.float64) @ blender_to_cv_local
    camera_from_world = np.linalg.inv(world_from_cv_camera)
    rotation_world_to_camera = camera_from_world[:3, :3]
    translation_world_to_camera = camera_from_world[:3, 3].reshape(3, 1)
    rotation_vector, _ = cv2.Rodrigues(rotation_world_to_camera)
    distortion = np.zeros((1, 5), dtype=np.float64)
    camera_matrix = np.asarray(
        [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(
        points_world_m.reshape(-1, 1, 3).astype(np.float64),
        rotation_vector,
        translation_world_to_camera,
        camera_matrix,
        distortion,
    )
    return projected.reshape(-1, 2)


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
    return (border_mean > inner_mean + min_border_delta) and (inner_std > min_inner_std)


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
        verify_modes: str = "warp_contrast,corner_subpix",
        warp_canonical_size: int = 48,
        warp_min_border_delta: float = 6.0,
        warp_min_inner_std: float = 7.0,
        subpix_window_half_size: int = 5,
        subpix_max_iterations: int = 40,
        subpix_epsilon: float = 0.01,
        contour_max_mean_error_px: float = 18.0,
        contrast_gate_min_range: float = 18.0,
        contrast_gate_patch_radius_px: int = 7,
        enable_photometric_refine: bool = False,
        pose_source: str = "oracle_center_world",
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
        self.contour_max_mean_error_px = float(contour_max_mean_error_px)
        self.contrast_gate_min_range = float(contrast_gate_min_range)
        self.contrast_gate_patch_radius_px = int(contrast_gate_patch_radius_px)
        self.enable_photometric_refine = bool(enable_photometric_refine)
        if pose_source not in ("oracle_center_world", "solvepnp_corners"):
            raise ValueError("pose_source must be oracle_center_world or solvepnp_corners")
        self.pose_source = str(pose_source)
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

    def prepare_frame(
        self,
        intrinsics: CameraIntrinsics,
        all_tags: Sequence[GroundTruthTag],
        camera_matrix_world: np.ndarray,
        *,
        sequence: str | None = None,
    ) -> None:
        """Caches oracle pose and layout for ROI projection; resets per-sequence state."""
        if sequence is not None and sequence != self._last_sequence:
            self._last_sequence = sequence
            self._sequence_requires_full_pupil = True
        self._intrinsics = intrinsics
        self._camera_matrix_world = np.asarray(camera_matrix_world, dtype=np.float64).reshape(4, 4)
        self._layout_tags = list(all_tags)

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
            "oracle_pose_upper_bound": 1,
            "no_bitwise_id_decode": 1,
            "photometric_refine_enabled": int(self.enable_photometric_refine),
            "photometric_extra_iterations_total": int(self._photometric_iterations_total),
            "pose_source": self.pose_source,
        }

    def detect(self, image_bgr: np.ndarray, intrinsics: CameraIntrinsics, tag_size_m: float) -> list[TagDetection]:
        """Runs full-image pupil on sequence start or fallback; otherwise custom ROI verification."""
        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        camera_matrix = np.asarray(
            [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        if self._sequence_requires_full_pupil or self._intrinsics is None or self._camera_matrix_world is None:
            self._pupil_full_frame_calls += 1
            self._sequence_requires_full_pupil = False
            self._coverage_values.append(1.0)
            self._region_counts.append(1)
            return self._full_pupil.detect(image_bgr, intrinsics, tag_size_m)

        projected = self._projected_layout(width, height)
        per_tag_rois = [
            _axis_aligned_roi_from_quad(corners, width, height, self.padding_factor, self.min_region_size_px)
            for _family, _tag_id, corners in projected
        ]
        regions_for_coverage = (
            _merge_axis_aligned_regions(list(per_tag_rois)) if self.merge_overlapping_rois else list(per_tag_rois)
        )
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
            return self._full_pupil.detect(image_bgr, intrinsics, tag_size_m)

        self._custom_accept_frames += 1
        self._verified_tag_count_total += len(custom_detections)
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
            corners_world = np.asarray(tag.corners_world, dtype=np.float64).reshape(4, 3)
            corners_image = _world_points_to_image_pixels(corners_world, camera_matrix_world, intrinsics)
            if not np.all(np.isfinite(corners_image)):
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
