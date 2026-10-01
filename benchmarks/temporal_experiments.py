"""Small, reproducible CPU experiments for the production temporal graph."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2

from .__main__ import DETECTION_POLICIES, _calibration, _score_accuracy_record
from .dataset import REPOSITORY_ROOT, cache_path, load_manifest, verify_dataset
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
from .report import atomic_json, collect_provenance, file_sha256, json_value

DEFAULT_DATASET = REPOSITORY_ROOT / "EagleEye-current-benchmark-metadata_manifest.json"
DEFAULT_CACHE = REPOSITORY_ROOT / "benchmarks" / "cache"


def _parser() -> argparse.ArgumentParser:
    """Build the deliberately narrow temporal experiment CLI."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks.temporal_experiments")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "temporal.json")
    parser.add_argument("--subset")
    parser.add_argument("--clip", action="append", help="Clip ID (repeatable)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First reported frame; earlier frames are replayed",
    )
    parser.add_argument(
        "--frames-per-clip", type=int, help="Reported frames per clip from start-frame"
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Exclude first N reported frames from timing only",
    )
    parser.add_argument("--timeout", type=float, metavar="SECONDS")
    parser.add_argument("--opencv-threads", type=int)
    return parser


def _provisional_tags(by_tag: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return truth observations that can actually be provisional/eligible."""
    return [
        tag
        for tag in by_tag
        if bool(tag.get("eligible"))
        or "provisional" in str(tag.get("category", "")).lower()
    ]


def _operation_timings(profile: Any) -> dict[str, float]:
    """Extract only stable operation execution timings from a profile snapshot."""
    if not isinstance(profile, dict):
        return {}
    operations = profile.get("operations")
    if not isinstance(operations, dict):
        return {}
    return {
        str(name): float(row["execution_time_ms"])
        for name, row in operations.items()
        if isinstance(row, dict)
        and isinstance(row.get("execution_time_ms"), (int, float))
        and math.isfinite(float(row["execution_time_ms"]))
    }


def compact_record(record: dict[str, Any], clip: str) -> dict[str, Any]:
    """Reduce one officially scored replay record to a JSONL-safe frame row."""
    metrics = record["metrics"]
    detection = metrics["detection"]
    provisional = _provisional_tags(detection.get("by_tag", []))
    output = record.get("output") or {}
    detections = json_value(output.get("detections") or [])
    return {
        "clip": clip,
        "frame_index": record["frame_index"],
        "pipeline_ms": record.get("pipeline_duration_ns", 0) / 1_000_000,
        "decode_ms": record.get("decode_duration_ns", 0) / 1_000_000,
        "operation_ms": _operation_timings(output.get("profile")),
        "provisional_truth_tags": len(provisional),
        "provisional_matched": sum(bool(tag.get("detected")) for tag in provisional),
        "total_matched": len(detection.get("matches", [])),
        "corner_errors_px": detection.get("corner_errors_px", []),
        "truth_matches": [
            {
                "truth_index": match["truth_index"],
                "tag_id": detection["by_tag"][match["truth_index"]]["tag_id"],
                "corner_error_px": match["corner_error_px"],
            }
            for match in detection.get("matches", [])
        ],
        "unmatched_detections": [
            detections[index]
            for index in detection.get("unmatched_detection_indices", [])
        ],
        "detections": [
            {"tag_id": int(item["tag_id"]), "corners": item["corners"]}
            for item in detections
        ],
        "detected_ids": [
            int(item["tag_id"]) if isinstance(item, dict) else int(item.tag_id)
            for item in (output.get("detections") or [])
            if (isinstance(item, dict) and item.get("tag_id") is not None)
            or (
                not isinstance(item, dict) and getattr(item, "tag_id", None) is not None
            )
        ],
        "pose_available": bool(metrics.get("pose_available")),
        "robot_errors": metrics.get("robot_pose"),
        "camera_errors": metrics.get("camera_pose"),
        "camera_pose": json_value(output.get("camera_pose_raw_edn")),
        "failure": record.get("failure"),
        "tp": detection["tp"],
        "fp": detection["fp"],
        "fn": detection["fn"],
    }


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    """Return compact timing/error statistics with an explicit denominator."""
    data = sorted(float(value) for value in values)
    if not data:
        return {"denominator": 0, "mean": None, "p50": None, "p95": None, "p99": None}

    def percentile(percent: float) -> float:
        """Interpolate one percentile of the sorted sample."""
        index = (len(data) - 1) * percent / 100
        lower, upper = math.floor(index), math.ceil(index)
        return data[lower] + (data[upper] - data[lower]) * (index - lower)

    return {
        "denominator": len(data),
        "mean": sum(data) / len(data),
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def _reacquisition(rows: list[dict[str, Any]], key: str) -> list[dict[str, int | None]]:
    """Measure recovery after each contiguous interval without provisional truth."""
    result: list[dict[str, int | None]] = []
    index = 0
    while index < len(rows):
        if rows[index]["provisional_truth_tags"]:
            index += 1
            continue
        start = index
        while index < len(rows) and not rows[index]["provisional_truth_tags"]:
            index += 1
        if index == len(rows):
            result.append(
                {
                    "after_no_provisional_start": rows[start]["frame_index"],
                    "visibility_frame": None,
                    "recovered_frame": None,
                    "delay_frames": None,
                }
            )
            break
        visible_end = index
        while visible_end < len(rows) and rows[visible_end]["provisional_truth_tags"]:
            visible_end += 1
        recovered = next(
            (
                frame_index
                for frame_index in range(index, visible_end)
                if rows[frame_index][key]
            ),
            None,
        )
        result.append(
            {
                "after_no_provisional_start": rows[start]["frame_index"],
                "visibility_frame": rows[index]["frame_index"],
                "recovered_frame": (
                    None if recovered is None else rows[recovered]["frame_index"]
                ),
                "delay_frames": (
                    None
                    if recovered is None
                    else rows[recovered]["frame_index"] - rows[index]["frame_index"]
                ),
            }
        )
    return result


def longest_pose_gap(rows: Iterable[dict[str, Any]]) -> int:
    """Count the longest consecutive missing-pose run without crossing clip boundaries."""
    longest = current = 0
    previous: tuple[str, int] | None = None
    for row in rows:
        key = (row["clip"], row["frame_index"])
        if previous is None or key != (previous[0], previous[1] + 1):
            current = 0
        current = 0 if row["pose_available"] else current + 1
        longest = max(longest, current)
        previous = key
    return longest


def summarize(
    rows: list[dict[str, Any]],
    *,
    include_reacquisition: bool = True,
    timing_start: int = 0,
) -> dict[str, Any]:
    """Summarize compact rows; provisional matching is deliberately only a proxy."""
    groups = {
        "all": rows,
        "tag_present": [row for row in rows if row["provisional_truth_tags"]],
        "tag_free": [row for row in rows if not row["provisional_truth_tags"]],
    }

    def group(data: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate one visibility group."""
        tp, fp, fn = (sum(row[name] for row in data) for name in ("tp", "fp", "fn"))
        provisional = sum(row["provisional_truth_tags"] for row in data)
        matched = sum(row["provisional_matched"] for row in data)
        pose_available = sum(bool(row["pose_available"]) for row in data)
        errors = [row["robot_errors"] for row in data if row["robot_errors"]]
        timed = [row for row in data if row["frame_index"] >= timing_start]
        stage_names = {name for row in timed for name in row["operation_ms"]}
        return {
            "frames_denominator": len(data),
            "failure_count": sum(row["failure"] is not None for row in data),
            "pose_available": {
                "count": pose_available,
                "rate": pose_available / len(data) if data else None,
            },
            "longest_pose_gap_frames": longest_pose_gap(data),
            "official_detection": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "eligible_recall": tp / (tp + fn) if tp + fn else None,
            },
            "provisional_match_rate_proxy_not_recall": {
                "matched": matched,
                "truth_tags_denominator": provisional,
                "rate": matched / provisional if provisional else None,
            },
            "pipeline_ms": _stats(row["pipeline_ms"] for row in timed),
            "decode_ms": _stats(row["decode_ms"] for row in timed),
            "native_detector_ms": _stats(
                row["native_detector_ms"]
                for row in timed
                if row.get("native_detector_ms") is not None
            ),
            "native_detector_calls": sum(
                row.get("native_detector_calls", 0) for row in timed
            ),
            "corner_error_px": _stats(
                value for row in data for value in row.get("corner_errors_px", [])
            ),
            "operation_ms": {
                name: _stats(
                    row["operation_ms"][name]
                    for row in timed
                    if name in row["operation_ms"]
                )
                for name in sorted(stage_names)
            },
            "robot_errors": {
                name: _stats(error[name] for error in errors if name in error)
                for name in (
                    "translation_3d_m",
                    "translation_xy_m",
                    "rotation_rad",
                    "yaw_absolute_error_rad",
                )
            },
        }

    result: dict[str, Any] = {
        "groups": {name: group(data) for name, data in groups.items()}
    }
    if include_reacquisition:
        result["reacquisition"] = {
            "detection": _reacquisition(rows, "provisional_matched"),
            "pose": _reacquisition(rows, "pose_available"),
        }
    return result


class _TimedNativeDetector:
    """Benchmark-local proxy around a native detector, without global patching."""

    def __init__(self, detector: Any, samples: list[float]) -> None:
        """Keep the native instance and the current frame's timing collector."""
        self.detector = detector
        self.samples = samples

    def detect(self, *args: Any, **kwargs: Any) -> Any:
        """Time calls even when native detection raises or finds no tags."""
        started = time.perf_counter_ns()
        try:
            return self.detector.detect(*args, **kwargs)
        finally:
            self.samples.append((time.perf_counter_ns() - started) / 1_000_000)

    def __getattr__(self, name: str) -> Any:
        """Preserve the wrapped detector's inspection attributes."""
        return getattr(self.detector, name)


@contextmanager
def _native_detector_timings(
    pipeline: Any, samples: list[float]
) -> Iterator[dict[str, str]]:
    """Observe every native detect call, including fallback and skipped graph frames."""
    operation = pipeline.get_operation_by_uuid("bench-detect")
    if operation is None:
        raise ValueError("benchmark detector operation is missing")
    detector = operation.instance.detector
    fields = (
        "detector",
        "_full_frame_detector",
        "_large_roi_detector",
        "_small_roi_detector",
    )
    originals = {name: getattr(detector, name) for name in fields}
    libraries = {
        str(Path(native.libc._name).resolve())
        for native in originals.values()
        if native is not None
    }
    hashes = {path: file_sha256(path) for path in sorted(libraries)}
    try:
        for name, native in originals.items():
            if native is not None:
                setattr(detector, name, _TimedNativeDetector(native, samples))
        yield hashes
    finally:
        for name, native in originals.items():
            setattr(detector, name, native)


def _run(args: argparse.Namespace) -> int:
    """Run selected local clips sequentially, streaming rows as they complete."""
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if args.frames_per_clip is not None and args.frames_per_clip <= 0:
        raise ValueError("frames-per-clip must be greater than zero")
    if args.start_frame < 0 or args.warmup_frames < 0:
        raise ValueError("start-frame and warmup-frames must be nonnegative")
    config = args.config.resolve()
    if config.name != "temporal.json":
        raise ValueError("temporal experiments require a config named temporal.json")
    if args.opencv_threads is not None:
        if args.opencv_threads <= 0:
            raise ValueError("opencv-threads must be greater than zero")
        cv2.setNumThreads(args.opencv_threads)
    manifest_path = args.dataset.resolve()
    manifest = load_manifest(manifest_path)
    clips = (
        manifest.clips
        if args.subset is None
        else [clip for clip in manifest.clips if args.subset in clip.roles]
    )
    if args.clip:
        unknown = set(args.clip) - {clip.id for clip in manifest.clips}
        if unknown:
            raise ValueError(f"unknown clip IDs: {sorted(unknown)}")
        clips = [clip for clip in clips if clip.id in set(args.clip)]
    if not clips:
        raise ValueError(
            f"dataset contains no clips for subset {args.subset!r} and IDs {args.clip!r}"
        )
    if any(args.start_frame >= clip.frame_count for clip in clips):
        raise ValueError("start-frame must be inside each selected clip")
    if manifest.detection_policy_version not in DETECTION_POLICIES:
        raise ValueError(
            f"unsupported detection policy {manifest.detection_policy_version}"
        )
    cache_dir = args.cache_dir.resolve()
    assets = {asset.path: cache_path(cache_dir, asset) for asset in manifest.assets}
    verify_dataset(manifest.model_copy(update={"clips": clips}), cache_dir)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    import temporal_acceleration

    metadata = collect_provenance(manifest_path, {"temporal": config})
    metadata.update(
        {
            "config_path": str(config),
            "config_content": json.loads(config.read_text()),
            "config_sha256": file_sha256(config),
            "dataset_sha256": file_sha256(manifest_path),
            "subset": args.subset,
            "clips": [clip.id for clip in clips],
            "clip_frame_counts": {clip.id: clip.frame_count for clip in clips},
            "cache_dir": str(cache_dir),
            "start_frame": args.start_frame,
            "warmup_frames": args.warmup_frames,
            "timing_note": "Native calls include fallback and skipped graph frames; operation_ms only exists when the graph completes. No-call frames are excluded from native_detector_ms statistics. Warmup excludes timing only.",
            "frames_per_clip": args.frames_per_clip,
            "timeout_seconds": args.timeout,
            "opencv_threads": cv2.getNumThreads(),
            "native_library_sha256": {
                str(path): file_sha256(path)
                for path in Path(temporal_acceleration.__file__).parent.glob("*.so")
            },
            "source_sha256": {
                name: file_sha256(REPOSITORY_ROOT / name)
                for name in (
                    "src/main_operations/definitions/temporal_acceleration_preprocessor_rust.py",
                    "src/main_operations/modules/apriltags/apriltag_detector.py",
                    "src/main_operations/modules/apriltags/pnp_localization.py",
                )
            },
        }
    )
    atomic_json(output / "metadata.json", metadata)
    deadline = time.monotonic() + args.timeout if args.timeout else None
    clips_summary: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    timed_out = False
    with gzip.open(output / "frames.jsonl.gz", "wt", encoding="utf-8") as stream:
        for clip in clips:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
            rows: list[dict[str, Any]] = []
            samples: list[float] = []

            def persist(
                record: dict[str, Any],
                clip_id: str = clip.id,
                clip_rows: list[dict[str, Any]] = rows,
                samples: list[float] = samples,
            ) -> None:
                """Score and persist one replayed frame."""
                _score_accuracy_record(
                    record, DETECTION_POLICIES[manifest.detection_policy_version]
                )
                row = compact_record(record, clip_id)
                row["reported"] = row["frame_index"] >= args.start_frame
                row["timing_included"] = (
                    row["frame_index"] >= args.start_frame + args.warmup_frames
                )
                row["native_detector_calls"] = len(samples)
                row["native_detector_ms"] = sum(samples) if samples else None
                samples.clear()
                stream.write(
                    json.dumps(json_value(row), separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
                clip_rows.append(row)

            limit = (
                min(clip.frame_count, args.start_frame + args.frames_per_clip)
                if args.frames_per_clip is not None
                else clip.frame_count
            )
            with (
                build_pipeline(
                    config,
                    _calibration(assets[clip.calibration]),
                    manifest.mounting_transforms[clip.camera_id].model_dump(),
                    map_path=DEFAULT_MAP_PATH,
                    manager=(manager := ReplayCameraManager()),
                ) as pipeline,
                SequentialVideoDecoder(assets[clip.video]) as decoder,
                _native_detector_timings(pipeline, samples) as native_libraries,
            ):
                metadata["apriltag_library_sha256"] = native_libraries
                atomic_json(output / "metadata.json", metadata)
                run_accuracy(
                    itertools.islice(decoder, limit),
                    manager,
                    pipeline,
                    clip.frame_rate_num,
                    clip.frame_rate_den,
                    collect_pipeline_outputs,
                    itertools.islice(
                        stream_annotations(
                            assets[clip.ground_truth],
                            clip.frame_rate_num,
                            clip.frame_rate_den,
                        ),
                        limit,
                    ),
                    on_record=persist,
                    retain_records=False,
                    should_stop=(lambda: time.monotonic() >= deadline)
                    if deadline
                    else None,
                )
            expected_frames = limit
            clip_timed_out = (
                deadline is not None
                and time.monotonic() >= deadline
                and len(rows) < expected_frames
            )
            if any(row["failure"] is not None for row in rows):
                raise RuntimeError(f"clip {clip.id} contains a failed frame")
            if not clip_timed_out and len(rows) != expected_frames:
                raise ValueError(
                    f"clip {clip.id} decoded {len(rows)} frames, expected {expected_frames}"
                )
            selected = [row for row in rows if row["frame_index"] >= args.start_frame]
            all_rows.extend(selected)
            timed_out = timed_out or clip_timed_out
            clips_summary.append(
                {
                    "clip": clip.id,
                    "partial": (
                        clip_timed_out
                        or expected_frames < clip.frame_count
                        or args.start_frame > 0
                    ),
                    "replayed_frames": len(rows),
                    "reported_start_frame": args.start_frame,
                    "timing_start_frame": args.start_frame + args.warmup_frames,
                    **summarize(
                        selected, timing_start=args.start_frame + args.warmup_frames
                    ),
                }
            )
            if clip_timed_out:
                break
    atomic_json(
        output / "summary.json",
        {
            "schema_version": 1,
            "partial": timed_out
            or args.start_frame > 0
            or any(
                args.frames_per_clip is not None
                and args.start_frame + args.frames_per_clip < clip.frame_count
                for clip in clips
            ),
            "timed_out": timed_out,
            **summarize(
                all_rows,
                include_reacquisition=False,
                timing_start=args.start_frame + args.warmup_frames,
            ),
            "clips": clips_summary,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the temporal CPU experiment."""
    try:
        return _run(_parser().parse_args(argv))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"temporal experiment error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
