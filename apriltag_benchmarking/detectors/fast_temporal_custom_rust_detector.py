"""Rust-accelerated fast temporal custom detector (benchmark package).

The native extension ``apriltag_fast_temporal`` performs pinhole projection, ROI
selection, and the same ``warp_contrast`` gate as the Python implementation.
Optional ``corner_subpix`` and photometric refinement still run in OpenCV on
the CPU after Rust returns candidate quads.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from ..utils import GroundTruthTag
from .base import CameraIntrinsics, TagDetection
from .fast_temporal_custom_detector import (
    FastTemporalCustomAprilTagDetector,
    refine_corners_subpix,
    verify_local_contrast_gate,
)

try:
    from apriltag_fast_temporal import FastTemporalCustomRustCore as _RustCoreType

    _RUST_CORE_AVAILABLE = True
except ImportError:
    _RustCoreType = None
    _RUST_CORE_AVAILABLE = False


def rust_extension_available() -> bool:
    """Returns whether the ``apriltag_fast_temporal`` extension is importable."""
    return _RUST_CORE_AVAILABLE


class FastTemporalCustomRustAprilTagDetector(FastTemporalCustomAprilTagDetector):
    """Same benchmark semantics as :class:`FastTemporalCustomAprilTagDetector` with a Rust warp path."""

    name = "fast-temporal-custom-rust-apriltags"

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
        if not _RUST_CORE_AVAILABLE:
            raise ImportError(
                "apriltag_fast_temporal is not built. From the repo root run: "
                "env -u CONDA_PREFIX uv run maturin develop --release "
                "--manifest-path apriltag_benchmarking/fast_temporal_custom_rust/Cargo.toml"
            )
        super().__init__(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            padding_factor=padding_factor,
            max_regions=max_regions,
            min_region_size_px=min_region_size_px,
            merge_overlapping_rois=merge_overlapping_rois,
            min_detection_count=min_detection_count,
            verify_modes=verify_modes,
            warp_canonical_size=warp_canonical_size,
            warp_min_border_delta=warp_min_border_delta,
            warp_min_inner_std=warp_min_inner_std,
            subpix_window_half_size=subpix_window_half_size,
            subpix_max_iterations=subpix_max_iterations,
            subpix_epsilon=subpix_epsilon,
            contour_max_mean_error_px=contour_max_mean_error_px,
            contrast_gate_min_range=contrast_gate_min_range,
            contrast_gate_patch_radius_px=contrast_gate_patch_radius_px,
            enable_photometric_refine=enable_photometric_refine,
            pose_source=pose_source,
        )
        self._rust = _RustCoreType()

    def prepare_frame(
        self,
        intrinsics: CameraIntrinsics,
        all_tags: Sequence[GroundTruthTag],
        camera_matrix_world: np.ndarray,
        *,
        sequence: str | None = None,
    ) -> None:
        """Caches layout and pushes the same data to the Rust core each frame."""
        super().prepare_frame(intrinsics, all_tags, camera_matrix_world, sequence=sequence)
        self._rust.set_intrinsics(
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.cx),
            float(intrinsics.cy),
        )
        tag_families: list[str] = []
        tag_ids: list[int] = []
        corners_flat: list[float] = []
        for tag in self._layout_tags:
            if tag.corners_world is None:
                continue
            tag_families.append(str(tag.tag_family))
            tag_ids.append(int(tag.tag_id))
            corners_flat.extend(np.asarray(tag.corners_world, dtype=np.float64).reshape(12).tolist())
        self._rust.set_layout(tag_families, tag_ids, corners_flat)
        self._rust.set_pose_blender_row_major(
            np.asarray(camera_matrix_world, dtype=np.float64).reshape(16).tolist()
        )
        self._rust.set_roi_params(
            float(self.padding_factor),
            int(self.max_regions),
            int(self.min_region_size_px),
            bool(self.merge_overlapping_rois),
        )
        self._rust.set_warp_params(
            int(self.warp_canonical_size),
            float(self.warp_min_border_delta),
            float(self.warp_min_inner_std),
        )

    def acceleration_report(self) -> dict[str, float | int | str]:
        """Adds a flag indicating the Rust warp engine was used for verification."""
        report = super().acceleration_report()
        report["rust_warp_engine"] = 1
        return report

    def detect(self, image_bgr: np.ndarray, intrinsics: CameraIntrinsics, tag_size_m: float) -> list[TagDetection]:
        """Uses Rust for projection and warp verification; OpenCV for optional sub-pixel refinement."""
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

        gray_contiguous = np.ascontiguousarray(gray, dtype=np.uint8)
        coverage, region_count = self._rust.merged_roi_coverage_fraction(gray_contiguous)
        self._coverage_values.append(float(coverage))
        self._region_counts.append(float(region_count))

        rust_rows = self._rust.process_frame(gray_contiguous)
        modes = self.verify_modes
        use_subpix = "corner_subpix" in modes

        custom_detections: list[TagDetection] = []
        dedup_keys: set[tuple[str, int]] = set()
        for tag_family, tag_id, flat_corners in rust_rows:
            dedup_key = (str(tag_family), int(tag_id))
            if dedup_key in dedup_keys:
                continue
            corners_xy = np.asarray(flat_corners, dtype=np.float64).reshape(4, 2)
            if use_subpix:
                if "warp_contrast" not in modes:
                    if not verify_local_contrast_gate(
                        gray,
                        corners_xy,
                        self.contrast_gate_patch_radius_px,
                        self.contrast_gate_min_range,
                    ):
                        continue
                corners_xy = refine_corners_subpix(
                    gray,
                    corners_xy,
                    self.subpix_window_half_size,
                    self.subpix_max_iterations,
                    self.subpix_epsilon,
                )
            if self.enable_photometric_refine:
                corners_xy = refine_corners_subpix(
                    gray,
                    corners_xy,
                    self.subpix_window_half_size + 1,
                    max(10, self.subpix_max_iterations // 2),
                    self.subpix_epsilon * 0.5,
                )
                self._photometric_iterations_total += 1

            dedup_keys.add(dedup_key)
            pose_t = self._pose_translation_m(
                tag_family,
                tag_id,
                corners_xy,
                camera_matrix,
                dist_coeffs,
                tag_size_m,
            )
            custom_detections.append(
                TagDetection(
                    tag_family=str(tag_family),
                    tag_id=int(tag_id),
                    corners=corners_xy,
                    center=np.mean(corners_xy, axis=0),
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
            self._region_counts[-1] = 1.0
            return self._full_pupil.detect(image_bgr, intrinsics, tag_size_m)

        self._custom_accept_frames += 1
        self._verified_tag_count_total += len(custom_detections)
        return custom_detections
