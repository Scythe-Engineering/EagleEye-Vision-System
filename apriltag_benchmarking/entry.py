from __future__ import annotations

import argparse
import json
import time
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from line_profiler import profile

from .detectors.fast_temporal_custom_detector import FastTemporalCustomAprilTagDetector
from .detectors.fast_temporal_custom_rust_detector import (
    FastTemporalCustomRustAprilTagDetector,
)
from .detectors.pupil_detector import PupilAprilTagDetector
from .detectors.temporal_pupil_detector import TemporalPupilAprilTagDetector
from .utils import (
    BenchmarkSummary,
    iter_frames,
    iter_sequence_metadata,
    estimate_world_from_cv_camera_from_detections,
    match_by_family_id,
    read_image,
)


def _debug_draw_frame(
    image: np.ndarray,
    detections: list,
    rois: list[tuple[int, int, int, int]],
    output_path: Path,
    label: str,
    raw_rois: list[tuple[int, int, int, int]] | None = None,
) -> None:
    canvas = image.copy()
    for x, y, w, h in raw_rois or []:
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 255), 1)
    for x, y, w, h in rois:
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
    for det in detections:
        corners = np.asarray(det.corners, dtype=float).reshape(4, 2).astype(int)
        cv2.polylines(canvas, [corners], True, (0, 255, 0), 2)
        center = tuple(np.asarray(det.center, dtype=float).reshape(2).astype(int))
        cv2.circle(canvas, center, 4, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            f"{det.tag_family}:{det.tag_id}",
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "apriltag_benchmark_data"
DEFAULT_OUTPUT = PROJECT_ROOT / "apriltag_benchmarking" / "results.json"
DEFAULT_DETECTORS = [
    "pupil",
    "temporal-pupil",
    "fast-temporal-custom",
    "fast-temporal-custom-rust-tag-plane",
    "fast-temporal-custom-rust-corners",
    "fast-temporal-custom-rust-none",
]


@profile
def build_detector(name: str, args: argparse.Namespace):
    kwargs = dict(
        families=args.families,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        refine_edges=args.refine_edges,
        decode_sharpening=args.decode_sharpening,
    )
    if name == "pupil":
        return PupilAprilTagDetector(**kwargs)
    if name == "temporal-pupil":
        return TemporalPupilAprilTagDetector(
            **kwargs,
            padding_factor=args.temporal_padding_factor,
            max_regions=args.temporal_max_regions,
            min_region_size_px=args.temporal_min_region_size_px,
            merge_overlapping=not args.no_temporal_merge,
        )
    if name == "fast-temporal-custom":
        return FastTemporalCustomAprilTagDetector(
            **kwargs,
            padding_factor=args.fast_temporal_padding_factor,
            max_regions=args.fast_temporal_max_regions,
            min_region_size_px=args.fast_temporal_min_region_size_px,
            merge_overlapping_rois=not args.fast_temporal_no_merge,
            min_detection_count=(
                None
                if int(args.fast_temporal_min_detection_count) < 0
                else int(args.fast_temporal_min_detection_count)
            ),
            verify_modes=args.fast_temporal_verify_modes,
            warp_canonical_size=args.fast_temporal_warp_canonical_size,
            warp_min_border_delta=args.fast_temporal_warp_min_border_delta,
            warp_min_inner_std=args.fast_temporal_warp_min_inner_std,
            subpix_window_half_size=args.fast_temporal_subpix_window_half_size,
            subpix_max_iterations=args.fast_temporal_subpix_max_iterations,
            subpix_epsilon=args.fast_temporal_subpix_epsilon,
            subpix_max_shift_px=args.fast_temporal_subpix_max_shift_px,
            contour_max_mean_error_px=args.fast_temporal_contour_max_mean_error_px,
            contrast_gate_min_range=args.fast_temporal_contrast_gate_min_range,
            contrast_gate_patch_radius_px=args.fast_temporal_contrast_gate_patch_radius_px,
            enable_photometric_refine=args.fast_temporal_enable_photometric_refine,
            pose_source=args.fast_temporal_pose_source,
            optical_tracking_mode=args.fast_temporal_optical_tracking_mode,
            keyframe_interval=args.fast_temporal_keyframe_interval,
        )
    if name in (
        "fast-temporal-custom-rust",
        "fast-temporal-custom-rust-tag-plane",
        "fast-temporal-custom-rust-corners",
        "fast-temporal-custom-rust-none",
    ):
        rust_mode = args.fast_temporal_optical_tracking_mode
        if name == "fast-temporal-custom-rust-tag-plane":
            rust_mode = "tag_plane"
        elif name == "fast-temporal-custom-rust-corners":
            rust_mode = "corners"
        elif name == "fast-temporal-custom-rust-none":
            rust_mode = "none"
        return FastTemporalCustomRustAprilTagDetector(
            **kwargs,
            padding_factor=args.fast_temporal_padding_factor,
            max_regions=args.fast_temporal_max_regions,
            min_region_size_px=args.fast_temporal_min_region_size_px,
            merge_overlapping_rois=not args.fast_temporal_no_merge,
            min_detection_count=(
                None
                if int(args.fast_temporal_min_detection_count) < 0
                else int(args.fast_temporal_min_detection_count)
            ),
            verify_modes=args.fast_temporal_verify_modes,
            warp_canonical_size=args.fast_temporal_warp_canonical_size,
            warp_min_border_delta=args.fast_temporal_warp_min_border_delta,
            warp_min_inner_std=args.fast_temporal_warp_min_inner_std,
            subpix_window_half_size=args.fast_temporal_subpix_window_half_size,
            subpix_max_iterations=args.fast_temporal_subpix_max_iterations,
            subpix_epsilon=args.fast_temporal_subpix_epsilon,
            subpix_max_shift_px=args.fast_temporal_subpix_max_shift_px,
            contour_max_mean_error_px=args.fast_temporal_contour_max_mean_error_px,
            contrast_gate_min_range=args.fast_temporal_contrast_gate_min_range,
            contrast_gate_patch_radius_px=args.fast_temporal_contrast_gate_patch_radius_px,
            enable_photometric_refine=args.fast_temporal_enable_photometric_refine,
            pose_source=args.fast_temporal_pose_source,
            optical_tracking_mode=rust_mode,
            keyframe_interval=args.fast_temporal_keyframe_interval,
        )
    raise ValueError(f"Unknown detector: {name}")


@profile
def output_path_for_detector(
    base_output: Path, detector_key: str, multiple_detectors: bool
) -> Path:
    if not multiple_detectors:
        return base_output
    return base_output.with_name(
        f"{base_output.stem}_{detector_key.replace('-', '_')}{base_output.suffix}"
    )


@profile
def count_frames(data_root: Path, max_frames: int | None = None) -> int:
    total = 0
    for _, meta in iter_sequence_metadata(data_root):
        frame_count = len(meta.get("frames", []))
        if max_frames is not None:
            frame_count = min(frame_count, max(0, max_frames - total))
        total += frame_count
        if max_frames is not None and total >= max_frames:
            break
    return total


@profile
def _load_image(path):
    return path, read_image(path)

def load_all_records(data_root: Path, max_frames: int | None = None) -> list:
    total_frames = count_frames(data_root, max_frames)
    records = list(iter_frames(data_root, max_frames=max_frames))
    print(f"Pre-loading {len(records)} frames into memory...")

    # Use multiprocessing to speed up image reading
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_load_image, r.image_path): r for r in records}
        image_dict = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading images", unit="img"):
            path, img = future.result()
            image_dict[path] = img

    for r in records:
        r.image_data = image_dict[r.image_path]

    return records


@profile
def run_benchmark(
    args: argparse.Namespace,
    records: list,
    detector_key: str | None = None,
    output_path: Path | None = None,
) -> dict:
    detector_key = detector_key or args.detector
    detector = build_detector(detector_key, args)
    summary = BenchmarkSummary(detector=detector.name)
    previous_world_from_cv_camera: np.ndarray | None = None
    previous_pose_sequence: str | None = None

    debug_dir = getattr(args, "debug_frames_dir", None)
    debug_limit = int(getattr(args, "debug_frames", 0) or 0)
    debug_written = 0

    try:
        for record_index, record in enumerate(tqdm(
            records,
            total=len(records) or None,
            desc=detector_key,
            unit="frame",
        )):
            image = record.image_data
            tag_size = args.tag_size_m or (
                record.tags[0].tag_size_m if record.tags else 0.24
            )

            if hasattr(detector, "prepare_frame"):
                detector.prepare_frame(
                    record.intrinsics,
                    record.all_tags,
                    None,
                    sequence=record.sequence,
                )

            start = time.perf_counter()
            detections = detector.detect(image, record.intrinsics, tag_size)
            elapsed = time.perf_counter() - start

            if debug_dir is not None and debug_written < debug_limit:
                rois = list(getattr(detector, "last_regions", []))
                _debug_draw_frame(
                    image,
                    detections,
                    rois,
                    Path(debug_dir) / f"{detector_key}_{record_index:04d}_{record.sequence}_f{record.frame}.jpg",
                    f"{detector_key} {record.sequence} frame {record.frame} det={len(detections)} roi={len(rois)}",
                    raw_rois=list(getattr(detector, "last_raw_regions", [])),
                )
                debug_written += 1

            if hasattr(detector, "update_pose_from_detections"):
                pose_guess = previous_world_from_cv_camera if previous_pose_sequence == record.sequence else None
                world_from_cv_camera = estimate_world_from_cv_camera_from_detections(
                    detections,
                    record.tags,
                    record.intrinsics,
                    pose_guess,
                    robust=not detector_key.startswith("fast-temporal-custom"),
                )
                detector.update_pose_from_detections(world_from_cv_camera, sequence=record.sequence)
                previous_world_from_cv_camera = world_from_cv_camera
                previous_pose_sequence = record.sequence if world_from_cv_camera is not None else None

            matches, missed, extras = match_by_family_id(record.tags, detections)
            summary.frames += 1
            summary.total_time_s += elapsed
            summary.detections += len(detections)
            summary.ground_truth_visible += len(record.tags)
            summary.true_positives += len(matches)
            summary.false_negatives += len(missed)
            summary.false_positives += len(extras)

            for gt, det in matches:
                summary.center_errors_px.append(
                    float(np.linalg.norm(det.center - gt.center_image_px))
                )
                summary.corner_errors_px.append(
                    float(
                        np.mean(
                            np.linalg.norm(det.corners - gt.corners_image_px, axis=1)
                        )
                    )
                )
                if det.pose_t is not None:
                    summary.pose_errors_m.append(
                        float(np.linalg.norm(det.pose_t - gt.position_camera_cv_m))
                    )

            if args.verbose:
                tqdm.write(
                    f"{record.sequence} frame {record.frame}: "
                    f"{len(detections)} detections, {len(matches)} TP, "
                    f"{len(missed)} FN, {len(extras)} FP, {elapsed * 1000:.2f} ms"
                )
    finally:
        if hasattr(detector, "close"):
            detector.close()

    result = summary.to_report()
    if hasattr(detector, "acceleration_report"):
        result["acceleration"] = detector.acceleration_report()

    report = {
        "data_root": str(args.data_root),
        "detector_key": detector_key,
        "output": str(output_path or args.output),
        "results": result,
    }
    output = output_path or args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AprilTag detectors on synthetic Blender metadata."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--detector",
        choices=["all", *DEFAULT_DETECTORS],
        default="all",
        help="Detector to run. Defaults to all implementations.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Limit frames per detector. Defaults to 200 to avoid a macOS pupil-apriltags native crash in temporal crop mode; use 0 for all frames.",
    )
    parser.add_argument(
        "--tag-size-m",
        type=float,
        default=None,
        help="Override metadata tag size for pose estimation.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--debug-frames",
        type=int,
        default=0,
        help="Write this many annotated frames per detector showing detections and ROIs.",
    )
    parser.add_argument(
        "--debug-frames-dir",
        type=Path,
        default=PROJECT_ROOT / "apriltag_benchmarking" / "debug_frames",
        help="Directory for --debug-frames annotated images.",
    )

    parser.add_argument("--families", default="tag36h11")
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--quad-decimate", type=float, default=1.0)
    parser.add_argument("--quad-sigma", type=float, default=0.8)
    parser.add_argument("--refine-edges", type=int, default=1)
    parser.add_argument("--decode-sharpening", type=float, default=0.25)

    parser.add_argument("--temporal-padding-factor", type=float, default=2.0)
    parser.add_argument("--temporal-max-regions", type=int, default=20)
    parser.add_argument("--temporal-min-region-size-px", type=int, default=24)
    parser.add_argument("--no-temporal-merge", action="store_true")

    parser.add_argument("--fast-temporal-padding-factor", type=float, default=2.0)
    parser.add_argument("--fast-temporal-max-regions", type=int, default=28)
    parser.add_argument("--fast-temporal-min-region-size-px", type=int, default=28)
    parser.add_argument("--fast-temporal-no-merge", action="store_true")
    parser.add_argument(
        "--fast-temporal-min-detection-count",
        type=int,
        default=1,
        help="Fallback to full-image pupil when custom verifications return fewer than this many tags. Use -1 to disable.",
    )
    parser.add_argument(
        "--fast-temporal-verify-modes",
        type=str,
        default="corner_subpix,warp_contrast",
        help="Comma-separated pipeline, e.g. warp_contrast,corner_subpix or corner_subpix or contour_quad,warp_contrast,corner_subpix.",
    )
    parser.add_argument("--fast-temporal-warp-canonical-size", type=int, default=48)
    parser.add_argument(
        "--fast-temporal-warp-min-border-delta", type=float, default=6.0
    )
    parser.add_argument("--fast-temporal-warp-min-inner-std", type=float, default=7.0)
    parser.add_argument("--fast-temporal-subpix-window-half-size", type=int, default=5)
    parser.add_argument("--fast-temporal-subpix-max-iterations", type=int, default=40)
    parser.add_argument("--fast-temporal-subpix-epsilon", type=float, default=0.01)
    parser.add_argument("--fast-temporal-subpix-max-shift-px", type=float, default=4.0)
    parser.add_argument(
        "--fast-temporal-contour-max-mean-error-px", type=float, default=18.0
    )
    parser.add_argument(
        "--fast-temporal-contrast-gate-min-range", type=float, default=18.0
    )
    parser.add_argument(
        "--fast-temporal-contrast-gate-patch-radius-px", type=int, default=7
    )
    parser.add_argument(
        "--fast-temporal-enable-photometric-refine", action="store_true"
    )
    parser.add_argument(
        "--fast-temporal-pose-source",
        choices=["oracle_center_world", "solvepnp_corners"],
        default="solvepnp_corners",
        help="oracle_center_world matches benchmark GT pose definition; solvepnp_corners uses cv2.solvePnP on refined corners.",
    )
    parser.add_argument(
        "--fast-temporal-optical-tracking-mode",
        choices=["tag_plane", "corners", "none"],
        default="tag_plane",
        help="Prediction source between detector keyframes: tag-plane LK homography, 4-corner LK, or pose projection only.",
    )
    parser.add_argument(
        "--fast-temporal-keyframe-interval",
        type=int,
        default=10,
        help="Run full-frame pupil as a detector keyframe after this many custom frames. Defaults to 10; use 0 to disable periodic keyframes.",
    )
    return parser.parse_args()


def print_report(report: dict) -> None:
    results = report["results"]
    print(f"Detector: {results['detector']}")
    print(
        f"Frames: {results['frames']} | FPS: {results['fps']:.2f} | avg ms/frame: {results['avg_ms_per_frame']:.2f}"
    )
    print(f"Precision: {results['precision']:.3f} | Recall: {results['recall']:.3f}")
    print(f"Pose error mean (m): {results['pose_error_m']['mean']}")
    if "acceleration" in results:
        accel = results["acceleration"]
        coverage_line = (
            f"ROI coverage: {accel['mean_roi_coverage'] * 100:.1f}% mean image area, "
            f"{accel['mean_regions']:.1f} regions/frame"
        )
        fallback_frames = accel.get("fallback_frames")
        if fallback_frames is not None:
            coverage_line += f", pupil full-frame calls {accel.get('pupil_full_frame_calls', 'n/a')}, fallback {fallback_frames}"
        print(coverage_line)
    print(f"Wrote: {report.get('output', DEFAULT_OUTPUT)}")


def print_comparison(reports: list[dict]) -> None:
    if len(reports) <= 1:
        return
    print("\n=== Detector comparison ===")
    print(
        f"{'detector':28} {'fps':>9} {'ms/frame':>10} {'precision':>10} {'recall':>8} {'pose mean m':>12}"
    )
    for report in reports:
        result = report["results"]
        pose_mean = result["pose_error_m"]["mean"]
        pose_text = "n/a" if pose_mean is None else f"{pose_mean:.4f}"
        print(
            f"{result['detector'][:28]:28} "
            f"{result['fps']:9.2f} "
            f"{result['avg_ms_per_frame']:10.2f} "
            f"{result['precision']:10.3f} "
            f"{result['recall']:8.3f} "
            f"{pose_text:>12}"
        )


@profile
def main() -> None:
    args = parse_args()
    if args.max_frames == 0:
        args.max_frames = None

    args.data_root = args.data_root.expanduser().resolve()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")
    if not any(args.data_root.glob("seq_*/metadata.json")):
        raise FileNotFoundError(
            f"No sequence metadata found under {args.data_root}. "
            "Expected files like seq_0000/metadata.json."
        )

    # Pre-load records once for all detectors
    records = load_all_records(args.data_root, args.max_frames)

    detector_keys = DEFAULT_DETECTORS if args.detector == "all" else [args.detector]
    multiple = len(detector_keys) > 1
    reports = []

    for detector_key in detector_keys:
        print(f"\n=== Running {detector_key} ===")
        report = run_benchmark(
            args,
            records,
            detector_key=detector_key,
            output_path=output_path_for_detector(args.output, detector_key, multiple),
        )
        reports.append(report)
        print_report(report)

    if multiple:
        combined = {
            "data_root": str(args.data_root),
            "output": str(args.output),
            "detectors": [report["results"] for report in reports],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(combined, file, indent=2)
        print(f"\nWrote combined report: {args.output}")
    print_comparison(reports)


if __name__ == "__main__":
    main()
