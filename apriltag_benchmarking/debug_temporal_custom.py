from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detectors.fast_temporal_custom_detector import FastTemporalCustomAprilTagDetector
from .utils import (
    FrameRecord,
    estimate_world_from_cv_camera_from_detections,
    iter_frames,
    match_by_family_id,
    read_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "apriltag_benchmark_data"
WINDOW_NAME = "Temporal custom AprilTag debugger"


@dataclass
class DebugResult:
    image: np.ndarray
    detections: list
    rois: list[tuple[int, int, int, int]]
    raw_rois: list[tuple[int, int, int, int]]
    projected_tags: list[tuple[str, int, np.ndarray]]
    world_from_cv_camera: np.ndarray | None
    full_frame_call_count: int
    fallback_frame_count: int
    custom_accept_count: int
    optical_tracking_mode: str


def build_detector(args: argparse.Namespace) -> FastTemporalCustomAprilTagDetector:
    return FastTemporalCustomAprilTagDetector(
        families=args.families,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        refine_edges=args.refine_edges,
        decode_sharpening=args.decode_sharpening,
        padding_factor=args.padding_factor,
        max_regions=args.max_regions,
        min_region_size_px=args.min_region_size_px,
        merge_overlapping_rois=not args.no_merge,
        min_detection_count=None if args.min_detection_count < 0 else args.min_detection_count,
        verify_modes=args.verify_modes,
        warp_canonical_size=args.warp_canonical_size,
        warp_min_border_delta=args.warp_min_border_delta,
        warp_min_inner_std=args.warp_min_inner_std,
        subpix_window_half_size=args.subpix_window_half_size,
        subpix_max_iterations=args.subpix_max_iterations,
        subpix_epsilon=args.subpix_epsilon,
        subpix_max_shift_px=args.subpix_max_shift_px,
        contour_max_mean_error_px=args.contour_max_mean_error_px,
        contrast_gate_min_range=args.contrast_gate_min_range,
        contrast_gate_patch_radius_px=args.contrast_gate_patch_radius_px,
        enable_photometric_refine=args.enable_photometric_refine,
        pose_source=args.pose_source,
        optical_tracking_mode=args.optical_tracking_mode,
        keyframe_interval=args.keyframe_interval,
    )


def run_detector_frame(
    detector: FastTemporalCustomAprilTagDetector,
    record: FrameRecord,
    previous_world_from_cv_camera: np.ndarray | None,
    previous_pose_sequence: str | None,
    tag_size_override: float | None,
) -> DebugResult:
    image = read_image(record.image_path)
    tag_size = tag_size_override or (record.tags[0].tag_size_m if record.tags else 0.24)
    detector.prepare_frame(record.intrinsics, record.all_tags, None, sequence=record.sequence)
    detections = detector.detect(image, record.intrinsics, tag_size)
    pose_guess = previous_world_from_cv_camera if previous_pose_sequence == record.sequence else None
    world_from_cv_camera = estimate_world_from_cv_camera_from_detections(
        detections,
        record.tags,
        record.intrinsics,
        pose_guess,
        robust=False,
    )
    detector.update_pose_from_detections(world_from_cv_camera, sequence=record.sequence)
    return DebugResult(
        image=image,
        detections=copy.deepcopy(detections),
        rois=list(getattr(detector, "last_regions", [])),
        raw_rois=list(getattr(detector, "last_raw_regions", [])),
        projected_tags=[
            (str(family), int(tag_id), np.asarray(corners, dtype=float).reshape(4, 2).copy())
            for family, tag_id, corners in getattr(detector, "last_projected_tags", [])
        ],
        world_from_cv_camera=None if world_from_cv_camera is None else world_from_cv_camera.copy(),
        full_frame_call_count=int(getattr(detector, "_pupil_full_frame_calls", 0)),
        fallback_frame_count=int(getattr(detector, "_fallback_frames", 0)),
        custom_accept_count=int(getattr(detector, "_custom_accept_frames", 0)),
        optical_tracking_mode=str(getattr(detector, "optical_tracking_mode", "tag_plane")),
    )


def draw_debug(record: FrameRecord, result: DebugResult, index: int, total: int) -> np.ndarray:
    canvas = result.image.copy()
    for x, y, w, h in result.raw_rois:
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 255), 1)
    for x, y, w, h in result.rois:
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
    for family, tag_id, projected_corners in result.projected_tags:
        corners_f = np.asarray(projected_corners, dtype=float).reshape(4, 2)
        corners = corners_f.astype(np.int32)
        cv2.polylines(canvas, [corners], True, (255, 128, 0), 2, cv2.LINE_AA)
        for corner_i, (u, v) in enumerate(corners):
            cv2.drawMarker(
                canvas,
                (int(u), int(v)),
                (255, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=13,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.putText(canvas, f"p{corner_i}", (int(u) + 5, int(v) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        center = tuple(np.mean(corners_f, axis=0).astype(int))
        cv2.putText(canvas, f"pred {family}:{tag_id}", (center[0] + 6, center[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 128, 0), 1, cv2.LINE_AA)

    for det in result.detections:
        corners = np.asarray(det.corners, dtype=float).reshape(4, 2).astype(np.int32)
        cv2.polylines(canvas, [corners], True, (0, 255, 0), 2)
        for corner_i, (u, v) in enumerate(corners):
            cv2.circle(canvas, (int(u), int(v)), 4, (0, 128 + 30 * corner_i, 255), -1)
            cv2.putText(canvas, str(corner_i), (int(u) + 3, int(v) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        center = tuple(np.asarray(det.center, dtype=float).reshape(2).astype(int))
        cv2.circle(canvas, center, 4, (0, 0, 255), -1)
        cv2.putText(canvas, f"{det.tag_family}:{det.tag_id}", (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    matches, missed, extras = match_by_family_id(record.tags, result.detections)
    lines = [
        f"{index + 1}/{total}  {record.sequence} frame={record.frame}  det={len(result.detections)} match={len(matches)} miss={len(missed)} extra={len(extras)}",
        f"ROI merged={len(result.rois)} raw={len(result.raw_rois)} predicted_tags={len(result.projected_tags)}  full_calls={result.full_frame_call_count} fallback={result.fallback_frame_count} custom={result.custom_accept_count}  optical={result.optical_tracking_mode}",
        "Legend: green=detected, cyan crosses/orange quad=predicted, yellow=merged ROI, magenta=raw ROI",
        "Keys: Left/Right scrub, O cycle optical tracking, S save PNG, Q/Esc quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive debugger for the Python custom temporal AprilTag detector.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sequence", help="Optional sequence directory name, e.g. seq_0001")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=Path("apriltag_benchmarking/debug_frames"))
    parser.add_argument("--tag-size-m", type=float, default=None)
    parser.add_argument("--families", default="tag36h11")
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--quad-decimate", type=float, default=1.0)
    parser.add_argument("--quad-sigma", type=float, default=0.8)
    parser.add_argument("--refine-edges", type=int, default=1)
    parser.add_argument("--decode-sharpening", type=float, default=0.25)
    parser.add_argument("--padding-factor", type=float, default=2.0)
    parser.add_argument("--max-regions", type=int, default=24)
    parser.add_argument("--min-region-size-px", type=int, default=28)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--min-detection-count", type=int, default=1, help="Use -1 to disable full-frame fallback on low count")
    parser.add_argument("--verify-modes", default="corner_subpix,warp_contrast")
    parser.add_argument("--warp-canonical-size", type=int, default=48)
    parser.add_argument("--warp-min-border-delta", type=float, default=6.0)
    parser.add_argument("--warp-min-inner-std", type=float, default=7.0)
    parser.add_argument("--subpix-window-half-size", type=int, default=5)
    parser.add_argument("--subpix-max-iterations", type=int, default=40)
    parser.add_argument("--subpix-epsilon", type=float, default=0.01)
    parser.add_argument("--subpix-max-shift-px", type=float, default=4.0)
    parser.add_argument("--contour-max-mean-error-px", type=float, default=18.0)
    parser.add_argument("--contrast-gate-min-range", type=float, default=18.0)
    parser.add_argument("--contrast-gate-patch-radius-px", type=int, default=7)
    parser.add_argument("--enable-photometric-refine", action="store_true")
    parser.add_argument("--pose-source", choices=["solvepnp_corners", "oracle_center_world"], default="solvepnp_corners")
    parser.add_argument("--optical-tracking-mode", choices=["tag_plane", "corners", "none"], default="tag_plane")
    parser.add_argument("--keyframe-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_frames = None if args.max_frames == 0 else args.max_frames
    records = [r for r in iter_frames(args.data_root, max_frames=max_frames) if args.sequence is None or r.sequence == args.sequence]
    if not records:
        raise SystemExit("No frames matched the requested data root/sequence/max-frames")

    detector = build_detector(args)
    optical_modes = ["tag_plane", "corners", "none"]
    if args.optical_tracking_mode not in optical_modes:
        raise SystemExit(f"Unknown optical tracking mode: {args.optical_tracking_mode}")
    detector.optical_tracking_mode = args.optical_tracking_mode
    cache: dict[int, DebugResult] = {}
    previous_world_from_cv_camera: np.ndarray | None = None
    previous_pose_sequence: str | None = None
    highest_computed = -1
    current = max(0, min(args.start_index, len(records) - 1))

    try:
        while highest_computed < current:
            highest_computed += 1
            res = run_detector_frame(detector, records[highest_computed], previous_world_from_cv_camera, previous_pose_sequence, args.tag_size_m)
            cache[highest_computed] = res
            previous_world_from_cv_camera = res.world_from_cv_camera
            previous_pose_sequence = records[highest_computed].sequence if res.world_from_cv_camera is not None else None

        while True:
            canvas = draw_debug(records[current], cache[current], current, len(records))
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKeyEx(0)
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                out = args.save_dir / f"custom_temporal_{current:04d}_{records[current].sequence}_f{records[current].frame}.png"
                cv2.imwrite(str(out), canvas)
                print(f"saved {out}")
            elif key in (ord("o"), ord("O")):
                mode_i = (optical_modes.index(detector.optical_tracking_mode) + 1) % len(optical_modes)
                args.optical_tracking_mode = optical_modes[mode_i]
                detector.close()
                detector = build_detector(args)
                cache.clear()
                previous_world_from_cv_camera = None
                previous_pose_sequence = None
                highest_computed = -1
                while highest_computed < current:
                    highest_computed += 1
                    res = run_detector_frame(
                        detector,
                        records[highest_computed],
                        previous_world_from_cv_camera,
                        previous_pose_sequence,
                        args.tag_size_m,
                    )
                    cache[highest_computed] = res
                    previous_world_from_cv_camera = res.world_from_cv_camera
                    previous_pose_sequence = records[highest_computed].sequence if res.world_from_cv_camera is not None else None
            elif key in (2424832, 81):  # left arrow on Win/Linux/OpenCV variants
                current = max(0, current - 1)
            elif key in (2555904, 83):  # right arrow
                if current + 1 < len(records):
                    current += 1
                    while highest_computed < current:
                        highest_computed += 1
                        res = run_detector_frame(detector, records[highest_computed], previous_world_from_cv_camera, previous_pose_sequence, args.tag_size_m)
                        cache[highest_computed] = res
                        previous_world_from_cv_camera = res.world_from_cv_camera
                        previous_pose_sequence = records[highest_computed].sequence if res.world_from_cv_camera is not None else None
    finally:
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
