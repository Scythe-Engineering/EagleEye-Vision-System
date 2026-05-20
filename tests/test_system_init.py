"""System initialization smoke tests without running the pipeline."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

from typing import Any, cast

import numpy as np
import pytest
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.logging.logger import Logger
from src.utils.timing import get_timing, unwrap_timed_deep
from tests.utils.dummy_dependencies import (
    DummyComputePool,
    FakeCameraThreadManager,
    FakeEagleEyeInterface,
    FakeNetworkTable,
    ReplayCameraThreadManager,
)
from tests.utils.dummy_data import (
    BenchmarkGroundTruthFrame,
    dummy_frame,
    load_benchmark_ground_truth_csv,
)


BENCHMARK_MANIFEST_RELATIVE_PATH = Path(
    "src/utils/sim_videos/benchmark_manifest.json"
)
SIM_VIDEO_RELATIVE_DIRECTORY = Path("src/utils/sim_videos")
DEFAULT_CAPTURE_PERIOD_US = 33_333


def test_pipeline_initialization_only(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    pipeline_config_path = project_root / "src" / "config" / "pipeline_config.json"
    with pipeline_config_path.open("r", encoding="utf-8") as handle:
        pipeline_config = json.load(handle)

    temp_config_path = tmp_path / "pipeline_config.json"
    temp_config_path.write_text(json.dumps(pipeline_config), encoding="utf-8")

    web_interface = FakeEagleEyeInterface()
    network_table = FakeNetworkTable()
    compute_pool = DummyComputePool()
    logger = Logger(log_directory="logs/test")
    camera_manager = FakeCameraThreadManager(default_frame=dummy_frame())
    camera_config_registry = CameraConfigRegistry()
    camera_manager.add_camera("basic_test")
    camera_manager.add_camera("FaceTime HD Camera")
    camera_manager.add_camera("test_camera")

    try:
        from src.config.utils.generate_all_pipelines import generate_all_pipelines
    except ImportError as exc:
        pytest.skip(f"system_init_optional: {exc}")

    pipelines = generate_all_pipelines(
        cast(Any, web_interface),
        cast(Any, compute_pool),
        cast(Any, network_table),
        cast(Any, camera_manager),
        camera_config_registry=camera_config_registry,
        logger=logger,
        pipeline_config=str(temp_config_path),
    )

    if not pipelines:
        pytest.skip("pipeline_init_optional: no pipelines created")

    for pipeline in pipelines.values():
        assert pipeline.operations, "Pipeline has no operations"
        assert pipeline.thread is None


def _project_root() -> Path:
    """Resolve the repository root for benchmark assets.

    Returns:
        Path: Repository root.
    """

    return Path(__file__).resolve().parents[1]


def _load_benchmark_specs(project_root: Path) -> list[dict[str, Any]]:
    """Load benchmark replay specifications from the asset manifest.

    Args:
        project_root: Repository root.

    Returns:
        List of benchmark replay specifications.
    """

    manifest_path = project_root / BENCHMARK_MANIFEST_RELATIVE_PATH
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    benchmarks = manifest.get("benchmarks", [])
    assert isinstance(benchmarks, list), "Benchmark manifest must contain a list"
    return cast(list[dict[str, Any]], benchmarks)


def _resolve_benchmark_asset_path(
    project_root: Path,
    relative_path_text: str,
    expected_suffix: str,
) -> Path:
    """Resolve and validate a benchmark asset path.

    Args:
        project_root: Repository root.
        relative_path_text: Project-relative asset path from the manifest.
        expected_suffix: Required file suffix.

    Returns:
        Path: Resolved asset path.
    """

    relative_path = Path(relative_path_text)
    assert not relative_path.is_absolute(), f"{relative_path_text} must be relative"
    assert (
        relative_path.suffix == expected_suffix
    ), f"{relative_path_text} must end with {expected_suffix}"

    asset_path = (project_root / relative_path).resolve()
    benchmark_directory = (project_root / SIM_VIDEO_RELATIVE_DIRECTORY).resolve()
    assert asset_path.is_relative_to(
        benchmark_directory
    ), f"{relative_path_text} must stay under {SIM_VIDEO_RELATIVE_DIRECTORY}"
    return asset_path


def test_benchmark_manifest_points_to_managed_assets() -> None:
    """Verify benchmark replay metadata uses the managed static asset layout."""

    project_root = _project_root()
    benchmark_specs = _load_benchmark_specs(project_root)
    assert benchmark_specs, "At least one benchmark replay spec is required"

    for benchmark_spec in benchmark_specs:
        video_path = _resolve_benchmark_asset_path(
            project_root,
            str(benchmark_spec["video_path"]),
            ".mp4",
        )
        ground_truth_path = _resolve_benchmark_asset_path(
            project_root,
            str(benchmark_spec["ground_truth_path"]),
            ".csv",
        )
        thresholds = benchmark_spec["thresholds"]

        assert video_path.name == f"{benchmark_spec['camera_bus_id']}.mp4"
        assert ground_truth_path.exists(), f"{ground_truth_path} is missing"
        assert thresholds["translation_rmse_m"] > 0
        assert thresholds["translation_max_error_m"] > 0
        assert thresholds["yaw_rmse_rad"] > 0
        assert thresholds["yaw_max_error_rad"] > 0
        assert thresholds["timestamp_max_drift_us"] >= 0
        assert thresholds["min_pose_samples"] > 0
        assert thresholds["min_apriltag_detection_frames"] >= 0


def _require_replay_dependencies() -> None:
    """Skip benchmark replay when native video or AprilTag dependencies are absent."""

    for module_name in ("cv2", "pupil_apriltags"):
        if importlib.util.find_spec(module_name) is None:
            pytest.skip(f"hardware_skip: {module_name} is unavailable")


def _decode_video_frames(video_path: Path, max_frames: int) -> list[np.ndarray]:
    """Decode benchmark video frames with OpenCV.

    Args:
        video_path: MP4 benchmark video path.
        max_frames: Maximum number of frames to decode.

    Returns:
        List of decoded BGR frames.
    """

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        pytest.skip(f"hardware_skip: {video_path.name} could not be opened")

    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        pytest.skip(f"hardware_skip: {video_path.name} has no decodable frames")
    return frames


def _pose_yaw_rad(pose_matrix: np.ndarray) -> float:
    """Calculate planar yaw from a 4x4 pose matrix.

    Args:
        pose_matrix: Robot pose transform.

    Returns:
        float: Planar yaw in radians.
    """

    return float(math.atan2(float(pose_matrix[1, 0]), float(pose_matrix[0, 0])))


def _wrapped_angle_error_rad(actual_yaw: float, expected_yaw: float) -> float:
    """Calculate wrapped absolute yaw error.

    Args:
        actual_yaw: Measured yaw in radians.
        expected_yaw: Expected yaw in radians.

    Returns:
        float: Absolute wrapped angle error in radians.
    """

    yaw_delta = actual_yaw - expected_yaw
    return abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))


def _get_operation_output(pipeline: Any, operation_name: str) -> Any:
    """Fetch the latest pipeline output for an operation name.

    Args:
        pipeline: Pipeline instance under test.
        operation_name: Operation action name without the .py suffix.

    Returns:
        Any: Latest operation output, or None when unavailable.
    """

    for operation_uuid, operation in pipeline.operations.items():
        if operation.name == operation_name:
            return pipeline.flow_manager.operation_outputs.get(operation_uuid)
    return None


def _get_detection_count(pipeline: Any) -> int:
    """Count AprilTag detections from the latest replay frame.

    Args:
        pipeline: Pipeline instance under test.

    Returns:
        int: Number of AprilTag detections.
    """

    detections = unwrap_timed_deep(_get_operation_output(pipeline, "detect_apriltags"))
    return len(detections) if isinstance(detections, list) else 0


def _compare_pose_to_ground_truth(
    pose_matrix: np.ndarray,
    expected_pose: BenchmarkGroundTruthFrame,
) -> tuple[float, float]:
    """Compare a measured robot pose with ground truth.

    Args:
        pose_matrix: Measured 4x4 robot pose transform.
        expected_pose: Expected pose annotation.

    Returns:
        tuple[float, float]: Translation and yaw errors.
    """

    measured_translation: np.ndarray = pose_matrix[:3, 3].astype(float)
    expected_translation: np.ndarray = np.array(
        [expected_pose.x, expected_pose.y, expected_pose.z],
        dtype=float,
    )
    translation_error = float(
        np.linalg.norm(measured_translation - expected_translation)
    )
    yaw_error = _wrapped_angle_error_rad(
        _pose_yaw_rad(pose_matrix),
        expected_pose.yaw_rad,
    )
    return translation_error, yaw_error


def _run_benchmark_replay(
    benchmark_spec: dict[str, Any],
    project_root: Path,
) -> bool:
    """Replay one benchmark video through the configured full pipeline.

    Args:
        benchmark_spec: Benchmark replay specification.
        project_root: Repository root.

    Returns:
        bool: True when the benchmark ran, False when its video is absent.
    """

    video_path = _resolve_benchmark_asset_path(
        project_root,
        str(benchmark_spec["video_path"]),
        ".mp4",
    )
    if not video_path.exists():
        return False

    _require_replay_dependencies()
    ground_truth_path = _resolve_benchmark_asset_path(
        project_root,
        str(benchmark_spec["ground_truth_path"]),
        ".csv",
    )
    ground_truth_by_frame = load_benchmark_ground_truth_csv(ground_truth_path)
    thresholds = benchmark_spec["thresholds"]
    max_frames = int(benchmark_spec.get("max_frames", len(ground_truth_by_frame)))
    frames = _decode_video_frames(video_path, max_frames)

    web_interface = FakeEagleEyeInterface()
    network_table = FakeNetworkTable()
    compute_pool = DummyComputePool()
    logger = Logger(log_directory="logs/test")
    camera_manager = ReplayCameraThreadManager(
        camera_name=str(benchmark_spec["camera_bus_id"]),
        frames=frames,
        capture_period_us=int(
            benchmark_spec.get("capture_period_us", DEFAULT_CAPTURE_PERIOD_US)
        ),
    )
    camera_config_registry = CameraConfigRegistry()

    from src.config.utils.generate_all_pipelines import generate_all_pipelines

    pipelines = generate_all_pipelines(
        cast(Any, web_interface),
        cast(Any, compute_pool),
        cast(Any, network_table),
        cast(Any, camera_manager),
        camera_config_registry=camera_config_registry,
        logger=logger,
        pipeline_config=str(project_root / "src" / "config" / "pipeline_config.json"),
    )
    pipeline = pipelines[str(benchmark_spec["pipeline_name"])]

    translation_errors: list[float] = []
    yaw_errors: list[float] = []
    timestamp_drifts_us: list[int] = []
    apriltag_detection_frames = 0

    for frame_index in range(len(frames)):
        if frame_index not in ground_truth_by_frame:
            continue

        camera_manager.advance_to_frame(frame_index)
        previous_pose_count = len(web_interface.robot_positions)
        pipeline.run()

        if _get_detection_count(pipeline) > 0:
            apriltag_detection_frames += 1

        if len(web_interface.robot_positions) == previous_pose_count:
            continue

        latest_pose = np.asarray(web_interface.robot_positions[-1], dtype=float)
        translation_error, yaw_error = _compare_pose_to_ground_truth(
            latest_pose,
            ground_truth_by_frame[frame_index],
        )
        translation_errors.append(translation_error)
        yaw_errors.append(yaw_error)

        robot_pose_output = _get_operation_output(pipeline, "robot_pose_output")
        timing = get_timing(robot_pose_output)
        if timing is not None:
            expected_capture_nt_us = frame_index * camera_manager.capture_period_us
            timestamp_drifts_us.append(
                abs(timing.capture_nt_us - expected_capture_nt_us)
            )

    assert len(translation_errors) >= thresholds["min_pose_samples"]
    assert apriltag_detection_frames >= thresholds["min_apriltag_detection_frames"]
    assert max(timestamp_drifts_us, default=0) <= thresholds["timestamp_max_drift_us"]
    assert max(translation_errors) <= thresholds["translation_max_error_m"]
    assert math.sqrt(float(np.mean(np.square(translation_errors)))) <= thresholds[
        "translation_rmse_m"
    ]
    assert max(yaw_errors) <= thresholds["yaw_max_error_rad"]
    assert math.sqrt(float(np.mean(np.square(yaw_errors)))) <= thresholds[
        "yaw_rmse_rad"
    ]
    return True


def test_benchmark_videos_replay_full_pipeline_against_ground_truth() -> None:
    """Replay available benchmark videos and enforce configured error margins."""

    project_root = _project_root()
    ran_benchmark = False
    missing_video_names: list[str] = []

    for benchmark_spec in _load_benchmark_specs(project_root):
        if _run_benchmark_replay(benchmark_spec, project_root):
            ran_benchmark = True
        else:
            missing_video_names.append(str(benchmark_spec["video_path"]))

    if not ran_benchmark:
        pytest.skip(
            "hardware_skip: benchmark video assets are unavailable: "
            + ", ".join(missing_video_names)
        )
