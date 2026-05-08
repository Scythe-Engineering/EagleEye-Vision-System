from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from pupil_apriltags import Detector

from .base import CameraIntrinsics, TagDetection
from .pupil_detector import PupilAprilTagDetector, suppress_native_stderr
from ..utils import GroundTruthTag

try:
    from temporal_acceleration import TemporalAcceleration as RustTemporalAcceleration
except Exception:  # pragma: no cover - fallback exercised only when extension is unavailable
    RustTemporalAcceleration = None


class TemporalPupilAprilTagDetector:
    """Pupil AprilTag detector restricted to temporal-predicted ROIs.

    The Rust temporal module expects a world_from_camera pose and known 3D tag
    layout. In this offline benchmark we seed it from the synthetic metadata so
    we can benchmark the preprocessing/ROI strategy independently of a robot pose
    estimator.
    """

    name = "temporal-pupil-apriltags"

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.8,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
        padding_factor: float = 2.0,
        max_regions: int = 20,
        min_region_size_px: int = 24,
        merge_overlapping: bool = True,
    ) -> None:
        self.base = PupilAprilTagDetector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
        )
        self._detector_kwargs = dict(
            families=families,
            nthreads=max(1, int(nthreads)),
            quad_decimate=max(1.0, float(quad_decimate)),
            quad_sigma=max(0.0, float(quad_sigma)),
            refine_edges=int(refine_edges),
            decode_sharpening=float(decode_sharpening),
        )
        self.crop_detector = self._new_crop_detector()
        self.families = families
        self.padding_factor = float(padding_factor)
        self.max_regions = int(max_regions)
        self.min_region_size_px = int(min_region_size_px)
        self.merge_overlapping = bool(merge_overlapping)
        self._accel = None
        self._layout_key: tuple[tuple[str, int], ...] | None = None
        self.last_regions: list[tuple[int, int, int, int]] = []
        self._coverage_values: list[float] = []
        self._region_counts: list[int] = []
        self._last_sequence: str | None = None
        self._has_pose_for_next_frame = False

    def prepare_frame(
        self,
        intrinsics: CameraIntrinsics,
        all_tags: Sequence[GroundTruthTag],
        camera_matrix_world: np.ndarray | None,
        *,
        sequence: str | None = None,
    ) -> None:
        if sequence is not None and sequence != self._last_sequence:
            self._last_sequence = sequence
            self._has_pose_for_next_frame = False
        self._ensure_accelerator(intrinsics, all_tags)

    def update_pose_from_detections(
        self,
        world_from_cv_camera: np.ndarray | None,
        *,
        sequence: str | None = None,
    ) -> None:
        if sequence is not None and sequence != self._last_sequence:
            self._last_sequence = sequence
            self._has_pose_for_next_frame = False
        if self._accel is None or world_from_cv_camera is None:
            self._has_pose_for_next_frame = False
            return
        self._accel.back_propagate_input(np.asarray(world_from_cv_camera, dtype=np.float32).reshape(-1).tolist())
        self._has_pose_for_next_frame = True

    def _ensure_accelerator(self, intrinsics: CameraIntrinsics, all_tags: Sequence[GroundTruthTag]) -> None:
        layout_key = tuple(sorted((tag.tag_family, tag.tag_id) for tag in all_tags))
        if self._accel is not None and layout_key == self._layout_key:
            return

        ids: list[int] = []
        corners_flat: list[float] = []
        centers_flat: list[float] = []
        for tag in sorted(all_tags, key=lambda t: (t.tag_family, t.tag_id)):
            if tag.corners_world is None or tag.center_world is None:
                continue
            ids.append(int(tag.tag_id))
            corners_flat.extend(np.asarray(tag.corners_world, dtype=np.float32).reshape(-1).tolist())
            centers_flat.extend(np.asarray(tag.center_world, dtype=np.float32).reshape(-1).tolist())

        camera_matrix = [
            float(intrinsics.fx), 0.0, float(intrinsics.cx),
            0.0, float(intrinsics.fy), float(intrinsics.cy),
            0.0, 0.0, 1.0,
        ]

        if RustTemporalAcceleration is not None and ids:
            self._accel = RustTemporalAcceleration(
                camera_matrix=camera_matrix,
                distortion_coefficients=[],
                apriltag_ids=ids,
                apriltag_corners=corners_flat,
                apriltag_centers=centers_flat,
                padding_factor=self.padding_factor,
                max_regions=self.max_regions,
                min_region_size_px=self.min_region_size_px,
            )
        else:
            self._accel = None
        self._layout_key = layout_key

    def detect(self, image_bgr: np.ndarray, intrinsics: CameraIntrinsics, tag_size_m: float) -> list[TagDetection]:
        height, width = image_bgr.shape[:2]
        regions = self._regions(width, height)
        self.last_regions = regions
        image_area = max(1, width * height)
        roi_area = sum(w * h for _x, _y, w, h in regions)
        self._coverage_values.append(min(1.0, roi_area / image_area))
        self._region_counts.append(len(regions))

        detections_by_key: dict[tuple[str, int], TagDetection] = {}
        camera_matrix = np.asarray(
            [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        dist_coeffs = np.zeros((4, 1), dtype=float)
        for x, y, w, h in regions:
            crop = image_bgr[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            with suppress_native_stderr(True):
                raw_detections = self.crop_detector.detect(gray, estimate_tag_pose=False)
            for raw in raw_detections:
                family = getattr(raw, "tag_family", self.families)
                if isinstance(family, bytes):
                    family = family.decode("utf-8")
                corners = np.asarray(raw.corners, dtype=float).reshape(4, 2) + np.asarray([x, y], dtype=float)
                center = np.asarray(raw.center, dtype=float).reshape(2) + np.asarray([x, y], dtype=float)
                pose_t = self._estimate_pose_t(corners, camera_matrix, dist_coeffs, tag_size_m)
                det = TagDetection(
                    tag_family=str(family),
                    tag_id=int(raw.tag_id),
                    corners=corners,
                    center=center,
                    pose_t=pose_t,
                    decision_margin=float(getattr(raw, "decision_margin", 0.0)),
                    hamming=int(getattr(raw, "hamming", 0)),
                )
                key = (det.tag_family, det.tag_id)
                old = detections_by_key.get(key)
                if old is None or (det.decision_margin or 0.0) > (old.decision_margin or 0.0):
                    detections_by_key[key] = det
        return list(detections_by_key.values())

    def _regions(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        if self._accel is None or not self._has_pose_for_next_frame:
            return [(0, 0, width, height)]
        _crops, raw_regions = self._accel.process_frame(width=width, height=height)
        regions = []
        for region in raw_regions:
            x1, y1, x2, y2 = [int(v) for v in region]
            x1 = max(0, min(width, x1)); x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1)); y2 = max(0, min(height, y2))
            w = x2 - x1
            h = y2 - y1
            # After clipping at image boundaries a predicted ROI can become a
            # very thin sliver (for example 1 px wide as a tag exits frame).
            # Passing those degenerate crops into pupil-apriltags can abort the
            # Python process from native code mid-benchmark.
            if w >= self.min_region_size_px and h >= self.min_region_size_px:
                regions.append((x1, y1, w, h))
        if not regions:
            return [(0, 0, width, height)]
        return self._merge_regions(regions) if self.merge_overlapping else regions

    def _merge_regions(self, regions: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        boxes = [[x, y, x + w, y + h] for x, y, w, h in regions]
        changed = True
        while changed:
            changed = False
            merged: list[list[int]] = []
            for box in boxes:
                for existing in merged:
                    if not (box[2] < existing[0] or box[0] > existing[2] or box[3] < existing[1] or box[1] > existing[3]):
                        existing[0] = min(existing[0], box[0]); existing[1] = min(existing[1], box[1])
                        existing[2] = max(existing[2], box[2]); existing[3] = max(existing[3], box[3])
                        changed = True
                        break
                else:
                    merged.append(box)
            boxes = merged
        return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]

    def _new_crop_detector(self) -> Detector:
        return Detector(**self._detector_kwargs)

    def _release_crop_detector(self) -> None:
        try:
            if hasattr(self.crop_detector, "tag_detector_ptr"):
                self.crop_detector.tag_detector_ptr = None
            if hasattr(self.crop_detector, "tag_families"):
                self.crop_detector.tag_families = {}
        except Exception:
            pass

    def _reset_crop_detector(self) -> None:
        self._release_crop_detector()
        self.crop_detector = self._new_crop_detector()

    def _estimate_pose_t(
        self,
        corners: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tag_size_m: float,
    ) -> np.ndarray | None:
        s = float(tag_size_m) * 0.78
        half = s * 0.5
        object_points = np.asarray(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=float,
        )
        ok, _rvec, tvec = cv2.solvePnP(
            object_points,
            corners.astype(float),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        return None if not ok else tvec.reshape(3)

    def acceleration_report(self) -> dict[str, float]:
        return {
            "mean_roi_coverage": float(np.mean(self._coverage_values)) if self._coverage_values else 1.0,
            "min_roi_coverage": float(np.min(self._coverage_values)) if self._coverage_values else 1.0,
            "max_roi_coverage": float(np.max(self._coverage_values)) if self._coverage_values else 1.0,
            "mean_regions": float(np.mean(self._region_counts)) if self._region_counts else 1.0,
        }

    def close(self) -> None:
        self.base.close()
        self._release_crop_detector()
