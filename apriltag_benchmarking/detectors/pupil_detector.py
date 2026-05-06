from __future__ import annotations

import contextlib
import os
import cv2
import numpy as np
from pupil_apriltags import Detector

from .base import AprilTagDetectorImplementation, CameraIntrinsics, TagDetection


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool = True):
    if not enabled:
        yield
        return
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


class PupilAprilTagDetector(AprilTagDetectorImplementation):
    name = "pupil-apriltags"

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
        quiet_native: bool = True,
    ) -> None:
        self.families = families
        self.quiet_native = quiet_native
        self.detector = Detector(
            families=families,
            nthreads=max(1, int(nthreads)),
            quad_decimate=max(1.0, float(quad_decimate)),
            quad_sigma=max(0.0, float(quad_sigma)),
            refine_edges=int(refine_edges),
            decode_sharpening=float(decode_sharpening),
        )

    def detect(
        self,
        image_bgr: np.ndarray,
        intrinsics: CameraIntrinsics,
        tag_size_m: float,
    ) -> list[TagDetection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        with suppress_native_stderr(self.quiet_native):
            raw = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=intrinsics.params,
                # pupil-apriltags defines tag_size as the black/white payload square
                # used by its detected corners. The synthetic generator's tag_size_m
                # is the full textured plane including the one-cell white border.
                # tag36h11 assets are 10x10 cells. Pupil's pose solver uses the
                # black border outer edge, which spans about 7.8/10 of this rendered
                # asset after Blender texture sampling/antialiasing.
                tag_size=float(tag_size_m) * 0.78,
            )
        detections: list[TagDetection] = []
        for det in raw:
            pose_t = None if getattr(det, "pose_t", None) is None else np.asarray(det.pose_t, dtype=float).reshape(3)
            family = getattr(det, "tag_family", self.families)
            if isinstance(family, bytes):
                family = family.decode("utf-8")
            detections.append(
                TagDetection(
                    tag_family=str(family),
                    tag_id=int(det.tag_id),
                    corners=np.asarray(det.corners, dtype=float).reshape(4, 2),
                    center=np.asarray(det.center, dtype=float).reshape(2),
                    pose_t=pose_t,
                    decision_margin=float(getattr(det, "decision_margin", 0.0)),
                    hamming=int(getattr(det, "hamming", 0)),
                )
            )
        return detections

    def close(self) -> None:
        try:
            if hasattr(self.detector, "tag_detector_ptr"):
                self.detector.tag_detector_ptr = None
            if hasattr(self.detector, "tag_families"):
                self.detector.tag_families = {}
        except Exception:
            pass
