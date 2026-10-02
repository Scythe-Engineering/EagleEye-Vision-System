"""Command-line interface for synthetic video benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dataset import (
    DEFAULT_VIDEO_ARCHIVE_URL,
    REPOSITORY_ROOT,
    cache_path,
    download_metadata,
    download_missing_assets,
    load_manifest,
    verify_dataset,
)
from .metrics import (
    Detection,
    TruthTag,
    availability,
    match_detections,
    pose_errors,
    summary_stats,
)
from .replay import (
    CONFIG_DIR,
    DEFAULT_MAP_PATH,
    ReplayCameraManager,
    SequentialVideoDecoder,
    build_pipeline,
    collect_pipeline_outputs,
    run_accuracy,
    stream_annotations,
)
from .report import (
    RunWriter,
    collect_provenance,
    file_sha256,
    read_frames,
    select_diagnostic_frame_ids,
    write_diagnostic_images,
)

DEFAULT_CACHE = REPOSITORY_ROOT / "benchmarks" / "cache"
DETECTION_POLICIES = {
    "pilot-candidate-v1": 8.0,
    "pilot-provisional-v1": 8.0,
}


def _parser() -> argparse.ArgumentParser:
    """Build the local accuracy benchmark command-line parser."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--subset")
    verify.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    verify.add_argument("--manifest-sha256")
    run = commands.add_parser("run")
    run.add_argument(
        "--dataset",
        type=Path,
        help="local manifest; defaults to manifest.json from the metadata ZIP",
    )
    run.add_argument("--subset")
    run.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    run.add_argument("--archive-url", default=DEFAULT_VIDEO_ARCHIVE_URL)
    run.add_argument("--manifest-sha256")
    run.add_argument(
        "--pipeline", choices=("both", "full-frame", "temporal"), default="both"
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="stop cleanly after this many seconds of elapsed time",
    )
    return parser


def _calibration(path: Path) -> dict[str, Any]:
    """Load a cached calibration in manifest or production key form."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if "camera_matrix" not in value and "matrix" in value:
        value["camera_matrix"] = value.pop("matrix")
    if "distortion_coefficients" not in value and "distortion" in value:
        value["distortion_coefficients"] = value.pop("distortion")
    return value


def _pipelines(selection: str) -> list[str]:
    """Resolve a CLI pipeline selection to run order."""
    return ["full-frame", "temporal"] if selection == "both" else [selection]


def _matrix4(value: Any) -> list[list[float]] | None:
    """Normalize a flat or nested serialized transform to a 4x4 list."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("matrix")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 16:
        value = [value[index : index + 4] for index in range(0, 16, 4)]
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        return None
    return [[float(item) for item in row] for row in value]


def _score_accuracy_record(
    record: dict[str, Any], tolerance_px: float
) -> dict[str, Any]:
    """Attach detection and pose errors under a versioned pixel tolerance."""
    truth = record.get("truth") or {}
    output = record.get("output") or {}
    truth_pairs = []
    for tag in truth.get("tags", []):
        corners = tag.get("corners", tag.get("corners_px"))
        if not isinstance(corners, list):
            continue
        truth_pairs.append(
            (
                tag,
                TruthTag(
                    int(tag["id"]),
                    tuple((float(point[0]), float(point[1])) for point in corners),
                    bool(tag.get("eligible", tag.get("eligibility") == "eligible")),
                    str(tag.get("eligibility", tag.get("category", "unclassified"))),
                ),
            )
        )
    truth_tags = [scored for _, scored in truth_pairs]
    detections = []
    for detection in output.get("detections") or []:
        tag_id = (
            detection.get("tag_id")
            if isinstance(detection, dict)
            else getattr(detection, "tag_id", None)
        )
        corners = (
            detection.get("corners")
            if isinstance(detection, dict)
            else getattr(detection, "corners", None)
        )
        if tag_id is None or corners is None:
            continue
        if hasattr(corners, "tolist"):
            corners = corners.tolist()
        detections.append(
            Detection(
                int(tag_id),
                tuple((float(point[0]), float(point[1])) for point in corners),
            )
        )
    detection_score = match_detections(
        detections, truth_tags, tolerance_px=tolerance_px
    )
    matched_truth = {item["truth_index"] for item in detection_score["matches"]}
    detection_score["by_tag"] = [
        {
            "tag_id": scored.tag_id,
            "projected_size_px": tag.get(
                "projected_min_edge_px", tag.get("projected_size")
            ),
            "eligible": scored.eligible,
            "category": scored.category,
            "detected": index in matched_truth,
        }
        for index, (tag, scored) in enumerate(truth_pairs)
    ]
    metrics: dict[str, Any] = {"detection": detection_score}

    camera_estimate = _matrix4(output.get("camera_pose_raw_edn"))
    camera_truth = _matrix4(truth.get("T_field_from_camera"))
    robot_estimate = _matrix4(output.get("robot_pose_nwu"))
    robot_truth = _matrix4(truth.get("T_field_from_robot"))
    metrics["camera_pose"] = (
        pose_errors(camera_estimate, camera_truth)
        if camera_estimate is not None and camera_truth is not None
        else None
    )
    metrics["robot_pose"] = (
        pose_errors(robot_estimate, robot_truth)
        if robot_estimate is not None and robot_truth is not None
        else None
    )
    metrics["pose_available"] = robot_estimate is not None
    record["metrics"] = metrics
    record["pose_available"] = metrics["pose_available"]
    return record


def _aggregate_accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate compact scored accuracy records with explicit denominators."""
    detection_rows = [record["metrics"]["detection"] for record in records]
    tp = sum(row["tp"] for row in detection_rows)
    fp = sum(row["fp"] for row in detection_rows)
    fn = sum(row["fn"] for row in detection_rows)
    corner_errors = [
        value for row in detection_rows for value in row["corner_errors_px"]
    ]
    robot_errors = [
        record["metrics"]["robot_pose"]
        for record in records
        if record["metrics"]["robot_pose"] is not None
    ]
    valid = [bool(record["metrics"]["pose_available"]) for record in records]
    timestamps = [int(record["timestamp_ns"]) for record in records]
    categories: dict[str, dict[str, int]] = {}
    for row in detection_rows:
        for tag in row.get("by_tag", []):
            counts = categories.setdefault(
                str(tag.get("category", "unclassified")),
                {"total": 0, "eligible": 0, "detected": 0},
            )
            counts["total"] += 1
            counts["eligible"] += int(bool(tag.get("eligible")))
            counts["detected"] += int(bool(tag.get("detected")))
    return {
        "frames": len(records),
        "detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "eligible_recall": tp / (tp + fn) if tp + fn else None,
            "corner_error_px": summary_stats(corner_errors),
            "by_category": categories,
        },
        "pose": {
            "translation_3d_m": summary_stats(
                row["translation_3d_m"] for row in robot_errors
            ),
            "translation_xy_m": summary_stats(
                row["translation_xy_m"] for row in robot_errors
            ),
            "rotation_rad": summary_stats(row["rotation_rad"] for row in robot_errors),
            "yaw_absolute_error_rad": summary_stats(
                row["yaw_absolute_error_rad"] for row in robot_errors
            ),
            "availability": availability(valid, timestamps),
        },
    }


def _run(args: argparse.Namespace) -> int:
    """Replay selected frames through production graphs and write accuracy results."""
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    deadline = time.monotonic() + args.timeout if args.timeout is not None else None
    dataset_path = args.dataset or download_metadata(
        args.archive_url, args.cache_dir
    )
    manifest = load_manifest(dataset_path, args.manifest_sha256)
    clips = (
        manifest.clips
        if args.subset is None
        else [c for c in manifest.clips if args.subset in c.roles]
    )
    if not clips:
        raise ValueError(f"dataset contains no clips for subset {args.subset!r}")
    if manifest.detection_policy_version not in DETECTION_POLICIES:
        raise ValueError(
            f"unsupported detection policy {manifest.detection_policy_version}"
        )
    download_missing_assets(manifest, args.cache_dir, args.archive_url, args.subset)
    verify_dataset(manifest, args.cache_dir, args.subset)
    configurations = _pipelines(args.pipeline)
    graph_paths = {
        name: CONFIG_DIR
        / ("full_frame.json" if name == "full-frame" else "temporal.json")
        for name in configurations
    }
    provenance = collect_provenance(dataset_path, graph_paths)
    assets = {
        asset.path: cache_path(args.cache_dir, asset) for asset in manifest.assets
    }
    metadata = {
        **provenance,
        "dataset_id": manifest.dataset_id,
        "dataset_release": manifest.release,
        "calibration_sha256": sorted(
            {file_sha256(assets[c.calibration]) for c in clips}
        ),
        "map_sha256": [file_sha256(DEFAULT_MAP_PATH)],
        "subset": args.subset,
        "pipeline": args.pipeline,
        "timeout_seconds": args.timeout,
    }
    writer = RunWriter.create(args.output, metadata, overwrite=args.overwrite)
    processing_started = time.monotonic()
    progress = tqdm(
        total=sum(clip.frame_count for clip in clips) * len(configurations),
        unit="frame",
        desc="Processing benchmark",
    )
    rows: list[dict[str, Any]] = []
    videos: dict[str, Path] = {}
    attempted = completed = failed = 0
    timed_out = False
    try:
        for clip in clips:
            videos[clip.id] = assets[clip.video]
            mounting = manifest.mounting_transforms[clip.camera_id].model_dump()
            for configuration in configurations:
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break
                base = {
                    "clip": clip.id,
                    "scenario": clip.scenario_id,
                    "variant": clip.variant_id,
                    "configuration": configuration,
                }
                compact: list[dict[str, Any]] = []

                def persist(
                    record: dict[str, Any],
                    base: dict[str, Any] = base,
                    compact: list[dict[str, Any]] = compact,
                ) -> None:
                    """Score and persist one replayed frame."""
                    nonlocal attempted, completed, failed
                    record.update(base)
                    _score_accuracy_record(
                        record, DETECTION_POLICIES[manifest.detection_policy_version]
                    )
                    writer.write_frame(record)
                    attempted += 1
                    completed += int(bool(record.get("completed")))
                    failed += int(bool(record.get("failure")))
                    progress.update(1)
                    compact.append(
                        {
                            "timestamp_ns": record["timestamp_ns"],
                            "metrics": record["metrics"],
                            "completed": bool(record.get("completed")),
                            "failure": bool(record.get("failure")),
                        }
                    )

                with (
                    build_pipeline(
                        graph_paths[configuration],
                        _calibration(assets[clip.calibration]),
                        mounting,
                        map_path=DEFAULT_MAP_PATH,
                        manager=(manager := ReplayCameraManager()),
                    ) as pipeline,
                    SequentialVideoDecoder(assets[clip.video]) as decoder,
                ):
                    run_accuracy(
                        decoder,
                        manager,
                        pipeline,
                        clip.frame_rate_num,
                        clip.frame_rate_den,
                        collect_pipeline_outputs,
                        stream_annotations(
                            assets[clip.ground_truth],
                            clip.frame_rate_num,
                            clip.frame_rate_den,
                        ),
                        on_record=persist,
                        retain_records=False,
                        should_stop=(
                            (lambda: time.monotonic() >= deadline)
                            if deadline is not None
                            else None
                        ),
                    )
                timed_out = deadline is not None and time.monotonic() >= deadline
                if not timed_out and len(compact) != clip.frame_count:
                    raise ValueError(
                        f"clip {clip.id} decoded {len(compact)} frames, manifest declares {clip.frame_count}"
                    )
                if not compact:
                    break
                rows.append(
                    {
                        **base,
                        "attempted": len(compact),
                        "completed": sum(r["completed"] for r in compact),
                        "failed": sum(r["failure"] for r in compact),
                        **_aggregate_accuracy(compact),
                    }
                )
                if timed_out:
                    break
            if timed_out:
                break
        total_tp = sum(r["detection"]["tp"] for r in rows)
        total_fp = sum(r["detection"]["fp"] for r in rows)
        total_fn = sum(r["detection"]["fn"] for r in rows)
        writer.finish_frames()
        diagnostic_ids = select_diagnostic_frame_ids(read_frames(args.output))
        images = write_diagnostic_images(args.output, read_frames(args.output), videos)
        summary = {
            "schema_version": 1,
            "attempted": attempted,
            "completed": completed,
            "skipped": attempted - completed - failed,
            "failed": failed,
            "timed_out": timed_out,
            "processing_seconds": time.monotonic() - processing_started,
            "detection": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": total_tp / (total_tp + total_fp)
                if total_tp + total_fp
                else None,
                "eligible_recall": total_tp / (total_tp + total_fn)
                if total_tp + total_fn
                else None,
            },
            "by_clip": rows,
            "diagnostic_frame_ids": diagnostic_ids,
            "diagnostic_images": images,
        }
        writer.finish(summary, rows)
        progress.close()
        print(f"Benchmark completed in {summary['processing_seconds']:.2f} seconds")
        return 2 if failed else 0
    finally:
        progress.close()
        writer.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Execute local verification or an accuracy run."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            manifest = load_manifest(args.manifest, args.manifest_sha256)
            verify_dataset(manifest, args.cache_dir, args.subset)
            print(f"verified {len(manifest.selected_assets(args.subset))} local assets")
            return 0
        return _run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
