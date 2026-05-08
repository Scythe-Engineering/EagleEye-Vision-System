from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from line_profiler import profile

from .detectors.base import CameraIntrinsics, TagDetection


@dataclass
class GroundTruthTag:
    tag_family: str
    tag_id: int
    tag_size_m: float
    corners_image_px: np.ndarray
    center_image_px: np.ndarray
    position_camera_cv_m: np.ndarray
    corners_world: np.ndarray | None = None
    center_world: np.ndarray | None = None


@dataclass
class FrameRecord:
    sequence: str
    frame: int
    image_path: Path
    intrinsics: CameraIntrinsics
    tags: list[GroundTruthTag]
    all_tags: list[GroundTruthTag]
    camera_matrix_world: np.ndarray


@dataclass
class BenchmarkSummary:
    detector: str
    frames: int = 0
    total_time_s: float = 0.0
    detections: int = 0
    ground_truth_visible: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    pose_errors_m: list[float] = field(default_factory=list)
    center_errors_px: list[float] = field(default_factory=list)
    corner_errors_px: list[float] = field(default_factory=list)

    def to_report(self) -> dict[str, Any]:
        def stats(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"mean": None, "median": None, "rmse": None, "max": None}
            arr = np.asarray(values, dtype=float)
            return {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "rmse": float(math.sqrt(np.mean(arr * arr))),
                "max": float(arr.max()),
            }

        fps = self.frames / self.total_time_s if self.total_time_s > 0 else 0.0
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives)
            else 0.0
        )
        recall = (
            self.true_positives / self.ground_truth_visible
            if self.ground_truth_visible
            else 0.0
        )
        return {
            **{
                k: v
                for k, v in asdict(self).items()
                if not k.endswith("_errors_m") and not k.endswith("_errors_px")
            },
            "fps": fps,
            "avg_ms_per_frame": (self.total_time_s / self.frames * 1000.0)
            if self.frames
            else 0.0,
            "precision": precision,
            "recall": recall,
            "pose_error_m": stats(self.pose_errors_m),
            "center_error_px": stats(self.center_errors_px),
            "corner_error_px": stats(self.corner_errors_px),
        }


@profile
def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@profile
def iter_sequence_metadata(data_root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    data_root = data_root.expanduser().resolve()
    for metadata_path in sorted(data_root.glob("seq_*/metadata.json")):
        yield metadata_path.parent, load_metadata(metadata_path)


@profile
def estimate_world_from_cv_camera_from_detections(
    detections: list[TagDetection],
    all_tags: list[GroundTruthTag],
    intrinsics: CameraIntrinsics,
    previous_world_from_cv_camera: np.ndarray | None = None,
    robust: bool = False,
) -> np.ndarray | None:
    """Estimate world-from-OpenCV-camera pose from current detections only."""
    tags_by_key = {(str(tag.tag_family), int(tag.tag_id)): tag for tag in all_tags}
    visible_by_key = {
        (str(tag.tag_family), int(tag.tag_id)): tag
        for tag in all_tags
        if np.all(np.isfinite(tag.corners_image_px))
    }
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for det in detections:
        key = (str(det.tag_family), int(det.tag_id))
        tag = tags_by_key.get(key)
        visible_tag = visible_by_key.get(key)
        if tag is None or tag.corners_world is None or visible_tag is None:
            continue
        center_image = np.asarray(visible_tag.center_image_px, dtype=np.float64).reshape(1, 2)
        # Metadata corners describe the full rendered 10x10 texture plane, while
        # pupil/custom detected corners describe the black-border payload square.
        # Validate against the same approximate payload-square model used for
        # solvePnP and temporal projection.
        visible_payload_corners = center_image + (
            np.asarray(visible_tag.corners_image_px, dtype=np.float64).reshape(4, 2) - center_image
        ) * 0.78
        det_corners = np.asarray(det.corners, dtype=float).reshape(4, 2)
        center_error = float(np.linalg.norm(np.asarray(det.center, dtype=float).reshape(2) - center_image.reshape(2)))
        corner_error = min(
            float(np.mean(np.linalg.norm(np.roll(det_corners, shift, axis=0) - visible_payload_corners, axis=1)))
            for shift in range(4)
        )
        if center_error > 45.0 or corner_error > 45.0:
            continue
        try:
            object_corners_full = np.asarray(tag.corners_world, dtype=np.float64).reshape(4, 3)
            if tag.center_world is not None:
                center_world = np.asarray(tag.center_world, dtype=np.float64).reshape(1, 3)
                # pupil-apriltags corners correspond to the black-border payload
                # square, not the full rendered 10x10 textured plane. Keep the
                # pose-update object model consistent with detected corners.
                object_corners = center_world + (object_corners_full - center_world) * 0.78
            else:
                object_corners = object_corners_full
            image_corners = np.asarray(det.corners, dtype=np.float64).reshape(4, 2)
        except (TypeError, ValueError):
            continue
        if not (np.all(np.isfinite(object_corners)) and np.all(np.isfinite(image_corners))):
            continue
        object_points.append(object_corners)
        image_points.append(image_corners)
    if not object_points:
        return None
    object_array = np.ascontiguousarray(np.concatenate(object_points, axis=0), dtype=np.float64).reshape(-1, 1, 3)
    image_array = np.ascontiguousarray(np.concatenate(image_points, axis=0), dtype=np.float64).reshape(-1, 1, 2)
    if object_array.shape[0] < 4 or image_array.shape[0] != object_array.shape[0]:
        return None
    camera_matrix = np.asarray(
        [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec_guess = None
    tvec_guess = None
    use_guess = previous_world_from_cv_camera is not None
    if use_guess:
        camera_from_world = np.linalg.inv(np.asarray(previous_world_from_cv_camera, dtype=np.float64).reshape(4, 4))
        rvec_guess, _ = cv2.Rodrigues(camera_from_world[:3, :3])
        tvec_guess = camera_from_world[:3, 3].reshape(3, 1)
    try:
        if robust:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_array,
                image_array,
                camera_matrix,
                dist_coeffs,
                rvec=rvec_guess,
                tvec=tvec_guess,
                useExtrinsicGuess=use_guess,
                iterationsCount=100,
                reprojectionError=8.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok or inliers is None or len(inliers) < 4:
                return None
            inlier_idx = inliers.reshape(-1)
            ok, rvec, tvec = cv2.solvePnP(
                object_array[inlier_idx],
                image_array[inlier_idx],
                camera_matrix,
                dist_coeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            inlier_idx = np.arange(object_array.shape[0])
            ok, rvec, tvec = cv2.solvePnP(
                object_array,
                image_array,
                camera_matrix,
                dist_coeffs,
                rvec=rvec_guess,
                tvec=tvec_guess,
                useExtrinsicGuess=use_guess,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
    except cv2.error:
        return None
    if not ok:
        return None
    projected, _ = cv2.projectPoints(object_array, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_array.reshape(-1, 2), axis=1)
    inlier_errors = errors[inlier_idx]
    if float(np.mean(inlier_errors)) > 12.0 or float(np.max(inlier_errors)) > 35.0:
        return None
    rotation_world_to_camera, _ = cv2.Rodrigues(rvec)
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = rotation_world_to_camera.T
    world_from_camera[:3, 3] = (-rotation_world_to_camera.T @ tvec.reshape(3, 1)).reshape(3)
    return world_from_camera


def blender_world_to_cv_camera(
    camera_matrix_world: list[list[float]], point_world: list[float]
) -> np.ndarray:
    camera_world = np.asarray(camera_matrix_world, dtype=float)
    world_point = np.asarray(
        [point_world[0], point_world[1], point_world[2], 1.0], dtype=float
    )
    point_blender_camera = np.linalg.inv(camera_world) @ world_point
    # Blender camera local: +X right, +Y up, camera looks down -Z. OpenCV: +X right, +Y down, +Z forward.
    return np.asarray(
        [point_blender_camera[0], -point_blender_camera[1], -point_blender_camera[2]],
        dtype=float,
    )


@profile
def iter_frames(
    data_root: Path, max_frames: int | None = None
) -> Iterable[FrameRecord]:
    yielded = 0
    for seq_dir, meta in iter_sequence_metadata(data_root):
        intr = CameraIntrinsics(**{k: float(v) for k, v in meta["intrinsics"].items()})
        for frame in meta.get("frames", []):
            tags: list[GroundTruthTag] = []
            all_tags: list[GroundTruthTag] = []
            for tag in frame.get("tags", []):
                corners = np.asarray(tag["corners_image_px"], dtype=float).reshape(4, 2)
                gt_tag = GroundTruthTag(
                    tag_family=str(tag["tag_family"]),
                    tag_id=int(tag["tag_id"]),
                    tag_size_m=float(tag["tag_size_m"]),
                    corners_image_px=corners,
                    center_image_px=np.asarray(
                        [
                            float(tag["center_ndc"][0]) * intr.width,
                            (1.0 - float(tag["center_ndc"][1])) * intr.height,
                        ],
                        dtype=float,
                    ),
                    position_camera_cv_m=blender_world_to_cv_camera(
                        frame["camera_matrix_world"], tag["center_world"]
                    ),
                    corners_world=np.asarray(tag["corners_world"], dtype=float).reshape(
                        4, 3
                    ),
                    center_world=np.asarray(tag["center_world"], dtype=float).reshape(
                        3
                    ),
                )
                all_tags.append(gt_tag)
                if tag.get("visible", False):
                    tags.append(gt_tag)
            image_path = Path(frame["image_path"])
            if not image_path.exists():
                # Blender's animation renderer may emit frame_0001.png even when
                # metadata was generated as frame_00001.png.
                alternate = seq_dir / "images" / f"frame_{int(frame['frame']):04d}.png"
                if alternate.exists():
                    image_path = alternate
            yield FrameRecord(
                seq_dir.name,
                int(frame["frame"]),
                image_path,
                intr,
                tags,
                all_tags,
                np.asarray(frame["camera_matrix_world"], dtype=float).reshape(4, 4),
            )
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                return


@profile
def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


@profile
def match_by_family_id(
    gt_tags: list[GroundTruthTag], detections: list[TagDetection]
) -> tuple[
    list[tuple[GroundTruthTag, TagDetection]], list[GroundTruthTag], list[TagDetection]
]:
    remaining = detections.copy()
    matches = []
    missed = []
    for gt in gt_tags:
        candidates = [
            d
            for d in remaining
            if d.tag_family == gt.tag_family and d.tag_id == gt.tag_id
        ]
        if not candidates:
            missed.append(gt)
            continue
        det = min(
            candidates,
            key=lambda d: float(np.linalg.norm(d.center - gt.center_image_px)),
        )
        remaining.remove(det)
        matches.append((gt, det))
    return matches, missed, remaining
