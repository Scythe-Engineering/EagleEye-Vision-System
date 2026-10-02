"""Causal artificial gyro replay through production operations."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.__main__ import _aggregate_accuracy, _parser, _score_accuracy_record
from benchmarks.gyro import GyroSettings, SyntheticGyro, truth_heading
from benchmarks.metrics import paired_pose_metrics
from benchmarks.replay import (
    CONFIG_DIR,
    DecodedFrame,
    ReplayCameraManager,
    ReplayError,
    benchmark_variant,
    build_pipeline,
    load_benchmark_config,
    run_accuracy,
)
from src.utils.timing import monotonic_ns_to_nt_us


def truth(yaw: float, position: int = 12345) -> dict[str, list[list[float]]]:
    """Build a field-from-robot pose with the requested yaw and position."""
    matrix = np.eye(4)
    matrix[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    matrix[:3, 3] = position
    return {"T_field_from_robot": matrix.tolist()}


def test_heading_noise_bias_wrap_delay_and_no_future_truth() -> None:
    """Replay deterministic noisy yaw with causal delayed delivery."""
    settings = GyroSettings(
        seed=17, noise_std_rad=0.01, bias_rad=0.1, delivery_delay_ms=15
    )
    gyro = SyntheticGyro(settings, 1_000_000_000)
    first = gyro.advance(truth(math.pi - 0.01), 0)
    assert first["delivered"] == []
    assert first["measurement"]["timestamp_us"] == monotonic_ns_to_nt_us(1_000_000_000)
    assert -math.pi <= first["measurement"]["value"] < math.pi
    second = gyro.advance(truth(-math.pi + 0.01), 10_000_000)
    assert second["delivered"] == []
    third = gyro.advance(truth(0), 20_000_000)
    assert [item["sample"] for item in third["delivered"]] == [first["measurement"]]
    assert all(
        item["sample"]["timestamp_us"] <= third["processing_timestamp_us"]
        for item in third["delivered"]
    )
    duplicate = SyntheticGyro(settings, 1_000_000_000)
    assert duplicate.advance(truth(math.pi - 0.01, -9876), 0) == first
    assert set(first["measurement"]) == {"timestamp_us", "value"}
    assert truth_heading(truth(0.7)) == pytest.approx(0.7)
    assert "NOT recorded physical gyro" in settings.provenance()["source"]
    with pytest.raises(ValueError, match="increase"):
        gyro.advance(truth(0), 20_000_000)


def test_processing_offset_and_oracle_are_explicit() -> None:
    """Validate explicit processing offsets and ideal-oracle settings."""
    gyro = SyntheticGyro(
        GyroSettings(delivery_delay_ms=15, processing_offset_ms=20), 1_000_000_000
    )
    record = gyro.advance(truth(0.2), 0)
    assert len(record["delivered"]) == 1
    assert record["delivered"][0]["sample"]["value"] == pytest.approx(0.2)
    assert GyroSettings(mode="ideal-oracle").provenance()["mode"] == "ideal-oracle"
    with pytest.raises(ValueError, match="ideal-oracle"):
        GyroSettings(mode="ideal-oracle", bias_rad=0.1)
    with pytest.raises(ValueError):
        GyroSettings(delivery_delay_ms=-1)
    with pytest.raises(ValueError, match="seed"):
        GyroSettings(seed=-1)


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("solver", ["normal", "2d"])
def test_real_variants_replay_causal_source_on_blank_frames(
    monkeypatch: pytest.MonkeyPatch, temporal: bool, solver: str
) -> None:
    """Drain causal gyro data once across paired production variants."""
    manager = ReplayCameraManager()
    calibration = {
        "camera_matrix": [[100, 0, 32], [0, 100, 32], [0, 0, 1]],
        "distortion_coefficients": [0] * 5,
    }
    annotations = [{"frame_index": i, **truth(0.3)} for i in range(3)]
    gyro = SyntheticGyro(GyroSettings(delivery_delay_ms=15), manager.epoch_ns)
    with build_pipeline(
        temporal,
        calibration,
        manager=manager,
        solver=solver,
        minimum_tags=2,
        paired=True,
    ) as pipeline:
        reader = pipeline.get_operation_by_uuid("bench-gyro").instance
        original_read = reader.run
        reads = []

        def read_once(value: Any) -> list[dict[str, Any]]:
            """Record each production queue drain for paired-replay assertions."""
            samples = original_read(value)
            reads.append(samples)
            return samples

        monkeypatch.setattr(reader, "run", read_once)
        frames = [DecodedFrame(i, np.zeros((64, 64, 3), np.uint8)) for i in range(3)]
        records = run_accuracy(
            frames, manager, pipeline, 100, aligned_truth=iter(annotations), gyro=gyro
        )
        assert pipeline.get_operation_errors() == []
        assert len(reads) == 3  # No second queue drain for the paired shadow.
        assert records[-1]["gyro"]["available_samples"] == reads[-1]
        assert records[0]["gyro"]["available_samples"] == []
        assert records[1]["gyro"]["available_samples"] == []
        assert records[2]["gyro"]["available_samples"] == [
            records[0]["gyro"]["measurement"]
        ]
        assert all(
            record["output"]["paired"]["rejection"] == "minimum_tags"
            for record in records
        )
        assert all(record["output"]["solver_duration_ns"] is None for record in records)
        for record in records:
            record["minimum_tags"] = 2
            _score_accuracy_record(record, 8)
        summary = _aggregate_accuracy(records)
        assert summary["identical_detector_outputs_paired"]["neither"] == 3
        assert summary["rejections"] == {"minimum_tags": 3}


def test_variant_keeps_detector_feedback_and_rejects_drift() -> None:
    """Preserve detector feedback and reject invalid variant wiring."""
    graph = load_benchmark_config(CONFIG_DIR / "temporal.json", True)
    variant = benchmark_variant(graph, "2d", 1)
    for uuid in ("bench-input", "bench-temporal", "bench-detect", "bench-robot"):
        assert next(node for node in graph if node["uuid"] == uuid) == next(
            node for node in variant if node["uuid"] == uuid
        )
    graph[2]["connections"][0]["to_port"] = "wrong"
    with pytest.raises(ReplayError, match="template ports"):
        benchmark_variant(graph, "2d", 1)
    args = _parser().parse_args(["run", "--output", "test"])
    assert (args.solver, args.minimum_tags, args.pipeline) == ("normal", "2", "both")


def test_paired_metrics_include_losses_gains_and_large_error_tails() -> None:
    """Include pose losses, gains, and large errors in paired metrics."""

    def row(index: int, error: float | None) -> dict[str, Any]:
        """Build a scored frame with optional pose availability."""
        return {
            "frame_index": index,
            "pose_available": error is not None,
            "robot_pose": None
            if error is None
            else {
                "translation_3d_m": error,
                "translation_xy_m": error,
                "rotation_rad": 0,
                "yaw_absolute_error_rad": 0,
            },
        }

    metrics = paired_pose_metrics(
        [row(0, 2), row(1, 0.1), row(2, None)], [row(0, 1.5), row(1, None), row(2, 0.2)]
    )
    assert (metrics["common_pose_frames"], metrics["lost"], metrics["gained"]) == (
        1,
        1,
        1,
    )
    assert metrics["large_xy_errors_over_1m"] == {"normal": 1, "2d": 1}
    assert metrics["large_3d_errors_over_1m"] == {"normal": 1, "2d": 1}
    assert (
        metrics["common_frame_error_delta_2d_minus_normal"]["translation_xy_m"]["mean"]
        == -0.5
    )
    assert metrics["common_frame_accuracy"]["normal"]["translation_3d_m"]["p99"] == 2
    assert (
        metrics["common_frame_error_delta_2d_minus_normal"]["translation_3d_m"][
            "median"
        ]
        == -0.5
    )


@pytest.mark.parametrize("delay_ms", [0, 100])
@pytest.mark.parametrize("minimum", [1, 2])
@pytest.mark.parametrize("solver", ["normal", "2d"])
def test_projected_detections_replay_real_solver_and_alignment(
    monkeypatch: pytest.MonkeyPatch, delay_ms: int, minimum: int, solver: str
) -> None:
    """Use production graph/NT/alignment with known distorted detector points."""
    from types import SimpleNamespace

    import cv2

    from src.main_operations.definitions import pnp_camera_localization as normal_module
    from src.main_operations.definitions import (
        pnp_camera_localization_2d as constrained_module,
    )
    from src.utils.camera_utils.camera_config_manager import CameraExtrinsics
    from src.utils.camera_utils.camera_coordinate_transforms import (
        build_robot_from_camera_transform,
    )
    from src.utils.timing import unwrap_timed

    matrix = np.array([[700.0, 0, 640], [0, 710, 400], [0, 0, 1]])
    distortion = np.array([-0.12, 0.03, 0.002, -0.001, 0.005])
    mounting = {
        "pitch": 13,
        "yaw": -17,
        "roll": 7,
        "x_offset": 0.3,
        "y_offset": -0.2,
        "z_offset": 0.7,
    }
    robot = np.eye(4)
    robot[:3, 3] = [3, 2, 0]
    camera = robot @ build_robot_from_camera_transform(CameraExtrinsics(**mounting))
    local = np.array(
        [
            [-0.6, -0.4, 4],
            [-0.2, -0.4, 4],
            [-0.2, 0, 4],
            [-0.6, 0, 4],
            [0.3, 0.1, 6],
            [0.7, 0.1, 6],
            [0.7, 0.5, 6],
            [0.3, 0.5, 6],
        ]
    )[: minimum * 4]
    points = local @ camera[:3, :3].T + camera[:3, 3]
    tags = {
        index + 1: SimpleNamespace(global_corners=corner)
        for index, corner in enumerate(points.reshape(-1, 4, 3))
    }
    monkeypatch.setattr(normal_module, "load_fmap_file", lambda path: tags)
    monkeypatch.setattr(constrained_module, "load_fmap_file", lambda path: tags)
    rotation = camera[:3, :3].T
    pixels = cv2.projectPoints(
        points,
        cv2.Rodrigues(rotation)[0],
        -rotation @ camera[:3, 3],
        matrix,
        distortion,
    )[0].reshape(-1, 4, 2)
    detections = [
        SimpleNamespace(tag_id=index + 1, corners=corner)
        for index, corner in enumerate(pixels)
    ]
    manager = ReplayCameraManager()
    calibration = {
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
    }
    with build_pipeline(
        False,
        calibration,
        mounting,
        manager=manager,
        solver=solver,
        minimum_tags=minimum,
        paired=True,
    ) as pipeline:
        # Deterministic detector output injection only; neither solver is mocked.
        monkeypatch.setattr(
            pipeline.get_operation_by_uuid("bench-detect").instance,
            "run",
            lambda frame: detections,
        )
        primary = pipeline.get_operation_by_uuid("bench-pnp").instance
        constrained = (
            primary if solver == "2d" else pipeline._benchmark_shadow.operation
        )
        assert isinstance(
            constrained, constrained_module.PnpCameraLocalization2DDefinition
        )
        original = constrained.run
        observed = []

        def spy(value: Any) -> dict[str, Any]:
            """Verify either constrained solver receives the cached gyro and detections."""
            assert set(value) == {"detections", "gyro_samples"}
            assert all(
                actual is expected
                for actual, expected in zip(
                    unwrap_timed(value["detections"]), detections
                )
            )
            assert value["gyro_samples"] == pipeline.get_operation_output(
                "bench-gyro", "data"
            )
            observed.append(value)
            return original(value)

        monkeypatch.setattr(constrained, "run", spy)
        annotation = {"frame_index": 0, "T_field_from_robot": robot.tolist()}
        records = run_accuracy(
            [DecodedFrame(0, np.zeros((800, 1280, 3), np.uint8))],
            manager,
            pipeline,
            100,
            aligned_truth=iter([annotation]),
            gyro=SyntheticGyro(
                GyroSettings(delivery_delay_ms=delay_ms), manager.epoch_ns
            ),
        )
        record = records[0]
        assert len(observed) == 1
        assert record["output"]["solver_duration_ns"] > 0
        record["minimum_tags"] = minimum
        _score_accuracy_record(record, 8)
        output = record["output"] if solver == "2d" else record["output"]["paired"]
        metrics = record["metrics"] if solver == "2d" else record["metrics"]["paired"]
        if delay_ms:
            assert output["diagnostics"]["reason"] == "missing_gyro"
            assert metrics["robot_pose"] is None
        else:
            assert output["diagnostics"]["alignment"] == "exact"
            assert metrics["robot_pose"]["translation_3d_m"] < 1e-7
        normal = record["metrics"]["paired"] if solver == "2d" else record["metrics"]
        assert normal["pose_available"]


def test_report_labels_oracle_and_keeps_solver_count_variants_separate(
    tmp_path: Path,
) -> None:
    """Label oracle provenance and keep solver/count reports distinct."""
    from benchmarks.report import RunWriter

    directory = tmp_path / "report"
    writer = RunWriter.create(
        directory, {"gyro": GyroSettings(mode="ideal-oracle").provenance()}
    )
    for config in ("temporal-normal-min1", "temporal-2d-min2"):
        writer.write_frame(
            {
                "clip": "test",
                "configuration": config,
                "frame_index": 0,
                "metrics": {
                    "pose_available": True,
                    "robot_pose": {"translation_3d_m": 0.1},
                },
            }
        )
    writer.finish({"partial": True}, [])
    report = (directory / "index.html").read_text()
    assert "Ideal oracle / oracle-equivalent" in report
    assert "NOT recorded physical gyro" in report
    assert "config=temporal-normal-min1" in report
    assert "config=temporal-2d-min2" in report
    assert '"seed": 0' in (directory / "run.json").read_text()


def test_native_publisher_keeps_delayed_measurement_timestamp() -> None:
    """The wire carries a double and the old measurement time, not send time."""
    from types import SimpleNamespace

    import ntcore

    instance = ntcore.NetworkTableInstance.create()
    instance.startLocal()
    topic = instance.getDoubleTopic("gyro")
    publisher = topic.publish(ntcore.PubSubOptions(sendAll=True, keepDuplicates=True))
    subscriber = topic.genericSubscribe(
        ntcore.PubSubOptions(sendAll=True, keepDuplicates=True)
    )
    try:
        manager = ReplayCameraManager()
        gyro = SyntheticGyro(GyroSettings(delivery_delay_ms=15), manager.epoch_ns)
        pipeline = SimpleNamespace(_benchmark_gyro_publisher=publisher)
        first = gyro.publish(pipeline, truth(0.25), 0)
        assert not any(value.isValid() for value in subscriber.readQueue())
        latest = gyro.publish(pipeline, truth(1.25), 20_000_000)
        queue = subscriber.readQueue()
        assert len(queue) == 1
        assert queue[0].value() == pytest.approx(0.25)
        assert queue[0].time() == first["measurement"]["timestamp_us"]
        assert queue[0].time() < latest["delivered"][0]["delivery_timestamp_us"]
    finally:
        del subscriber  # GenericSubscriber releases its handle on destruction.
        publisher.close()
        instance.stopLocal()
        ntcore.NetworkTableInstance.destroy(instance)


@pytest.mark.parametrize("shadow_solver", ["normal", "2d", None])
def test_identical_detection_comparison_keeps_failed_frames(
    shadow_solver: str | None,
) -> None:
    """Keep mixed and entirely failed clips in both solver populations."""
    records = [
        {
            "frame_index": index,
            "timestamp_ns": index * 10_000_000,
            "failure": "pipeline operation error",
            "output": None,
        }
        for index in range(2)
    ]
    if shadow_solver is not None:
        records.append(
            {
                "frame_index": 2,
                "timestamp_ns": 20_000_000,
                "failure": None,
                "output": {"paired": {"solver": shadow_solver}},
            }
        )
    for record in records:
        _score_accuracy_record(record, 8)
    result = _aggregate_accuracy(records, paired=True)[
        "identical_detector_outputs_paired"
    ]
    assert result["attempted_frames"] == {"normal": len(records), "2d": len(records)}
    assert result["failed_frames"] == {"normal": 2, "2d": 2}
    assert result["matched_frames"] == result["neither"] == len(records)
    assert result["matched_pose_availability"] == {"normal": 0, "2d": 0}
    if shadow_solver is None:
        assert "identical_detector_outputs_paired" not in _aggregate_accuracy(records)


def test_paired_failure_denominators_include_unmatched_attempts() -> None:
    """Unavailable and failed attempts must not disappear from the populations."""
    normal = [
        {"frame_index": 0, "pose_available": False, "failure": "reader"},
        {"frame_index": 1, "pose_available": False},
    ]
    constrained = [{"frame_index": 0, "pose_available": False}]
    result = paired_pose_metrics(normal, constrained)
    assert result["attempted_frames"] == {"normal": 2, "2d": 1}
    assert result["failed_frames"] == {"normal": 1, "2d": 0}
    assert result["unmatched_normal_frames"] == 1
    assert result["matched_pose_availability"] == {"normal": 0, "2d": 0}
    assert result["neither"] == 1


@pytest.mark.parametrize("change", ["gate", "feedback", "duplicate", "dangling"])
def test_variant_rejects_gate_feedback_and_identity_drift(change: str) -> None:
    """Strict preset validation includes the count gate and feedback branch."""
    graph = load_benchmark_config(CONFIG_DIR / "temporal.json", True)
    by_id = {node["uuid"]: node for node in graph}
    if change == "gate":
        by_id["bench-minimum"]["connections"][0]["to_port"] = "wrong"
    elif change == "feedback":
        feedback = next(
            edge
            for edge in by_id["bench-pnp"]["connections"]
            if edge["to_uuid"] == "bench-temporal"
        )
        feedback["is_default"] = not feedback["is_default"]
    elif change == "duplicate":
        graph.append(graph[0].copy())
    else:
        by_id["bench-minimum"]["connections"][0]["to_uuid"] = "missing"
    with pytest.raises(ReplayError):
        benchmark_variant(graph, "2d", 1)


@pytest.mark.parametrize("max_frames", [None, 2, 10])
def test_cli_expected_population_respects_frame_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_frames: int | None
) -> None:
    """Run the real CLI scoring/report path with a finite deterministic source."""
    import json
    from contextlib import nullcontext

    from benchmarks import __main__ as cli
    from tests.test_benchmark_dataset import _manifest

    calibration = {
        "camera_matrix": [[100, 0, 32], [0, 100, 32], [0, 0, 1]],
        "distortion_coefficients": [0] * 5,
    }
    payload = json.dumps(calibration).encode()
    manifest = _manifest(payload)
    asset = cli.cache_path(tmp_path, manifest.assets[0])
    asset.parent.mkdir(parents=True)
    asset.write_bytes(payload)
    manifest.clips[0].frame_count = 5
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    count = min(5, max_frames or 5)
    graph_path = CONFIG_DIR / "full_frame.json"
    graph = json.loads(graph_path.read_text())
    monkeypatch.setattr(cli, "download_missing_assets", lambda *args: None)
    monkeypatch.setattr(
        cli,
        "collect_provenance",
        lambda *args: {
            "graphs": {"full-frame": graph},
            "graph_sha256": {"full-frame": "base"},
        },
    )
    monkeypatch.setattr(
        cli, "build_resolved_pipeline", lambda *args, **kwargs: nullcontext(object())
    )
    monkeypatch.setattr(
        cli, "SequentialVideoDecoder", lambda *args: nullcontext(range(count))
    )
    monkeypatch.setattr(cli, "stream_annotations", lambda *args: iter(()))
    monkeypatch.setattr(cli, "select_diagnostic_frame_ids", lambda *args: [])
    monkeypatch.setattr(cli, "write_diagnostic_images", lambda *args: [])

    def replay(*args: Any, **kwargs: Any) -> None:
        assert kwargs["max_frames"] == max_frames
        for index in args[0]:
            kwargs["on_record"](
                {
                    "frame_index": index,
                    "timestamp_ns": index * 10_000_000,
                    "truth": {},
                    "output": {},
                    "completed": True,
                    "failure": None,
                }
            )

    monkeypatch.setattr(cli, "run_accuracy", replay)
    output = tmp_path / "report"
    arguments = [
        "run",
        "--dataset",
        str(manifest_path),
        "--cache-dir",
        str(tmp_path),
        "--pipeline",
        "full-frame",
        "--output",
        str(output),
    ]
    if max_frames is not None:
        arguments.extend(["--max-frames", str(max_frames)])
    assert cli._run(cli._parser().parse_args(arguments)) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["expected_selected_frames_per_variant"] == count
    assert summary["attempted"] == count
    assert summary["partial"] is (count < 5)


def test_failed_pipeline_is_not_reported_as_a_tag_gate_rejection() -> None:
    """Missing outputs on a failed attempt are not evidence of too few tags."""
    record = {
        "frame_index": 0,
        "timestamp_ns": 0,
        "truth": truth(0),
        "failure": "pipeline operation error",
        "output": None,
    }
    _score_accuracy_record(record, 8)
    summary = _aggregate_accuracy([record])
    assert summary["rejections"] == {"pipeline_error": 1}
    assert summary["pose"]["availability"]["availability"] == 0
