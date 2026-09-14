"""Finite, timestamp-correct replay through the production pipeline scheduler."""

from __future__ import annotations

import gzip
import json
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, TypeVar

import cv2
import ntcore
import numpy as np

from src.utils.camera_utils.camera_coordinate_transforms import (
    pose_local_edn_to_nwu,
)
from src.utils.timing import (
    FramePacket,
    TimedValue,
    TimingMetadata,
    get_timing,
    monotonic_ns_to_nt_us,
    unwrap_timed,
)

T = TypeVar("T")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
TEMPLATE_PATH = REPOSITORY_ROOT / "src/webui/js/pipeline/pipelineTemplates.json"
DEFAULT_MAP_PATH = (
    REPOSITORY_ROOT
    / "src/webui/assets/fields/2026/apriltag_maps/FE-2026-_REBUILTTM_Playing_Field.fmap"
)


class ReplayError(RuntimeError):
    """Raised when decoding, alignment, scheduling, or a pipeline cycle fails."""


class _HeadlessWeb:
    """No-op web callback target used by benchmark pipelines."""

    def __getattr__(self, _name: str) -> Callable[..., None]:
        """Return a no-op for optional UI callbacks."""
        return lambda *_args, **_kwargs: None


class _Logger:
    """Non-persistent logger accepted by production operations."""

    def log(self, _message: str) -> None:
        """Discard one benchmark diagnostic."""


class _ModelLibrary:
    """Non-persistent collaborator for graphs with no model operation."""


class PipelineLifecycle(AbstractContextManager[Any]):
    """Own a pipeline, local NT instance, and temporary calibration tree."""

    def __init__(
        self,
        pipeline: Any,
        nt_instance: Any,
        temporary: tempfile.TemporaryDirectory[str],
    ) -> None:
        """Record resources in reverse close order."""
        self.pipeline = pipeline
        self.nt_instance = nt_instance
        self.temporary = temporary
        self.closed = False

    def __enter__(self) -> Any:
        """Return the production pipeline."""
        return self.pipeline

    def close(self) -> None:
        """Close all owned resources once."""
        if self.closed:
            return
        self.closed = True
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            pipeline.close()
        instance, self.nt_instance = self.nt_instance, None
        if instance is not None:
            instance.stopLocal()
            ntcore.NetworkTableInstance.destroy(instance)
        self.temporary.cleanup()

    def __exit__(self, *_args: object) -> None:
        """Close resources when leaving the context."""
        self.close()


def _absolute_map_paths(
    config: list[dict[str, Any]], map_path: str | Path | None
) -> None:
    """Resolve every map-consuming operation independently of process cwd."""
    map_operations = {
        "pnp_camera_localization",
        "temporal_acceleration_preprocessor_rust",
    }
    configured = next(
        (
            node.get("action_params", {}).get("apriltag_map_path")
            for node in config
            if node.get("action_params", {}).get("apriltag_map_path")
        ),
        None,
    )
    selected = map_path or configured
    if selected is None:
        raise ReplayError("benchmark graph requires an apriltag map")
    path = Path(selected)
    absolute = path if path.is_absolute() else REPOSITORY_ROOT / path
    for node in config:
        name = str(node.get("action_name", "")).removesuffix(".py")
        if name in map_operations:
            node.setdefault("action_params", {})["apriltag_map_path"] = str(absolute)


def _normalize_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    """Normalize manifest and production calibration key names."""
    matrix = calibration.get("camera_matrix")
    distortion = calibration.get(
        "distortion_coefficients", calibration.get("distortion")
    )
    if matrix is None or distortion is None:
        raise ReplayError(
            "calibration requires camera_matrix and distortion coefficients"
        )
    array = np.asarray(matrix, dtype=float)
    coefficients = np.asarray(distortion, dtype=float).reshape(-1)
    if (
        array.shape != (3, 3)
        or not np.isfinite(array).all()
        or not np.isfinite(coefficients).all()
        or array[0, 0] <= 0
        or array[1, 1] <= 0
    ):
        raise ReplayError("calibration contains an invalid camera matrix")
    return {
        "camera_matrix": array.tolist(),
        "distortion_coefficients": coefficients.tolist(),
    }


def build_pipeline(
    config: str | Path | bool,
    calibration: str | Path | dict[str, Any],
    extrinsics: dict[str, Any] | None = None,
    *,
    map_path: str | Path | None = None,
    manager: ReplayCameraManager | None = None,
) -> PipelineLifecycle:
    """Construct the real production Pipeline with isolated collaborators.

    Args:
        config: Graph path, or False/True for full-frame/temporal defaults.
        calibration: Production intrinsics JSON path or equivalent mapping.
        extrinsics: Mounting values in production degrees/meters.
        map_path: Optional map override for every map-consuming operation.
        manager: Finite benchmark packet source.

    Returns:
        A context manager that owns the pipeline and temporary resources.
    """
    from src.config.utils.pipeline import Pipeline
    from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
    from src.utils.device_registry import DeviceDescriptor, DeviceRegistry

    if isinstance(config, bool):
        config_path = CONFIG_DIR / ("temporal.json" if config else "full_frame.json")
        expected_temporal = config
    else:
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = REPOSITORY_ROOT / config_path
        expected_temporal = config_path.stem == "temporal"
    graph = json.loads(
        json.dumps(load_benchmark_config(config_path, expected_temporal))
    )
    _absolute_map_paths(graph, map_path)

    if isinstance(calibration, dict):
        calibration_data = calibration
    else:
        calibration_path = Path(calibration)
        calibration_data = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_data = _normalize_calibration(calibration_data)
    mount = extrinsics or {
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "x_offset": 0.0,
        "y_offset": 0.0,
        "z_offset": 0.0,
    }

    temporary = tempfile.TemporaryDirectory(prefix="eagleeye-replay-")
    camera_dir = Path(temporary.name) / "benchmark-camera"
    camera_dir.mkdir()
    (camera_dir / "intrinsics.json").write_text(
        json.dumps(calibration_data), encoding="utf-8"
    )
    (camera_dir / "extrinsics.json").write_text(json.dumps(mount), encoding="utf-8")
    registry = CameraConfigRegistry(temporary.name)
    if registry.load_all_from_directory() != 1:
        temporary.cleanup()
        raise ReplayError("isolated camera calibration did not load")

    replay = manager or ReplayCameraManager()
    instance = ntcore.NetworkTableInstance.create()
    instance.startLocal()
    try:
        pipeline = Pipeline(
            graph,
            _HeadlessWeb(),  # type: ignore[arg-type]
            instance.getTable("benchmark"),
            _Logger(),  # type: ignore[arg-type]
            DeviceRegistry([DeviceDescriptor("cpu", "CPU", "cpu", None)]),
            _ModelLibrary(),  # type: ignore[arg-type]
            replay,  # type: ignore[arg-type]
            camera_config_registry=registry,
            camera_bus_ids=[replay.bus_id],
            pipeline_name="benchmark-replay",
            limit_frames_to_camera_capture_speed=False,
        )
        actual = {operation.name for operation in pipeline.operations.values()}
        expected = {str(node["action_name"]).removesuffix(".py") for node in graph}
        if actual != expected:
            raise ReplayError(
                "pipeline operation identities differ: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )
        if pipeline.get_operation_errors():
            raise ReplayError(
                f"pipeline initialization errors: {pipeline.get_operation_errors()}"
            )
    except BaseException:
        instance.stopLocal()
        ntcore.NetworkTableInstance.destroy(instance)
        temporary.cleanup()
        raise
    return PipelineLifecycle(pipeline, instance, temporary)


@dataclass(frozen=True)
class DecodedFrame:
    """One decoded image and its authoritative zero-based dataset index."""

    index: int
    image: Any
    decode_duration_ns: int = 0


class SequentialVideoDecoder:
    """Sequential OpenCV decoder with finite EOF and bounded lookahead."""

    def __init__(self, path: str | Path, max_lookahead: int = 8) -> None:
        """Open a video without decoding more than eight frames ahead."""
        if not 1 <= max_lookahead <= 8:
            raise ValueError("max_lookahead must be between one and eight")
        self.path = Path(path)
        self.max_lookahead = max_lookahead
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise ReplayError(f"OpenCV could not open video decoder for {self.path}")
        self._queue: deque[DecodedFrame] = deque()
        self._next_index = 0
        self.eof = False
        self.error: ReplayError | None = None

    def _fill(self) -> None:
        """Fill the bounded lookahead queue or establish terminal state."""
        while len(self._queue) < self.max_lookahead and not self.eof:
            started = time.perf_counter_ns()
            ok, image = self._capture.read()
            duration = time.perf_counter_ns() - started
            if ok and image is not None:
                self._queue.append(DecodedFrame(self._next_index, image, duration))
                self._next_index += 1
                continue
            advertised = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if advertised > 0 and self._next_index < advertised:
                self.error = ReplayError(
                    f"decoder stopped at frame {self._next_index}; expected {advertised}"
                )
            self.eof = True

    def read(self) -> DecodedFrame | None:
        """Return the next frame, None at clean EOF, or raise a decode error."""
        self._fill()
        if self._queue:
            item = self._queue.popleft()
            self._fill()
            return item
        if self.error is not None:
            raise self.error
        return None

    def __iter__(self) -> Iterator[DecodedFrame]:
        """Yield every frame exactly once in source order."""
        while (item := self.read()) is not None:
            yield item

    def close(self) -> None:
        """Release the decoder and discard bounded lookahead."""
        self._queue.clear()
        self._capture.release()

    def __enter__(self) -> Self:
        """Return this open decoder."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Release decoder resources."""
        self.close()


def rational_timestamp_ns(
    frame_index: int, fps_numerator: int, fps_denominator: int = 1
) -> int:
    """Return round(index / fps) in ns without cumulative period drift."""
    if frame_index < 0 or fps_numerator <= 0 or fps_denominator <= 0:
        raise ValueError("frame index must be nonnegative and frame rate positive")
    numerator = frame_index * 1_000_000_000 * fps_denominator
    quotient, remainder = divmod(numerator, fps_numerator)
    twice = remainder * 2
    return quotient + (
        twice > fps_numerator or (twice == fps_numerator and quotient % 2 == 1)
    )


class ReplayCameraManager:
    """Benchmark-local packet methods used by the real DeviceInput operation."""

    def __init__(
        self, bus_id: str = "benchmark-camera", epoch_ns: int | None = None
    ) -> None:
        """Create an empty finite source using one immutable timeline epoch."""
        self.bus_id = bus_id
        self.camera_name = "synthetic-benchmark"
        self.epoch_ns = time.monotonic_ns() if epoch_ns is None else epoch_ns
        self._current: FramePacket | None = None
        self._latched: FramePacket | None = None
        self._cycle_open = False
        self.eof = False
        self.error: BaseException | None = None

    def publish(self, frame: Any, frame_index: int, timestamp_ns: int) -> FramePacket:
        """Publish one immutable packet whose clock fields identify one instant."""
        if self.eof or self.error is not None:
            raise ReplayError("cannot publish after source termination")
        instant = self.epoch_ns + timestamp_ns
        packet = TimedValue(
            frame,
            TimingMetadata(
                monotonic_ns_to_nt_us(instant),
                instant,
                frame_seq=frame_index,
                camera_name=self.camera_name,
                bus_id=self.bus_id,
            ),
        )
        self._current = packet
        return packet

    def begin_cycle(self) -> FramePacket:
        """Latch the current packet so every read in this cycle is identical."""
        if self._cycle_open:
            raise ReplayError("replay cycle is already open")
        if self._current is None:
            raise ReplayError("no frame is available")
        self._latched = self._current
        self._cycle_open = True
        return self._latched

    def end_cycle(self) -> None:
        """Release the per-cycle latch."""
        self._latched = None
        self._cycle_open = False

    def get_current_packet_by_bus_id(self, bus_id: str) -> FramePacket | None:
        """Return the latched/current packet for the configured bus ID."""
        if bus_id != self.bus_id:
            return None
        return self._latched if self._cycle_open else self._current

    def get_current_timing_by_bus_id(self, bus_id: str) -> TimingMetadata | None:
        """Return current packet timing without copying its image."""
        packet = self.get_current_packet_by_bus_id(bus_id)
        return packet.timing if packet is not None else None

    def get_camera_name_by_bus_id(self, bus_id: str) -> str | None:
        """Resolve the sole synthetic camera name."""
        return self.camera_name if bus_id == self.bus_id else None

    def wait_for_new_frame_by_bus_id(
        self,
        bus_id: str,
        after_frame_seq: int,
        timeout_s: float | None = None,
    ) -> bool:
        """Report whether the finite source currently has a newer packet."""
        del timeout_s
        timing = self.get_current_timing_by_bus_id(bus_id)
        return (
            timing is not None
            and timing.frame_seq is not None
            and timing.frame_seq > after_frame_seq
        )

    def mark_eof(self) -> None:
        """Mark clean EOF without invalidating the final packet."""
        self.eof = True

    def mark_error(self, error: BaseException) -> None:
        """Mark a terminal source failure."""
        self.error = error


def stream_annotations(
    path: str | Path,
    fps_numerator: int,
    fps_denominator: int = 1,
) -> Iterator[dict[str, Any]]:
    """Stream JSONL or JSONL.gz while enforcing contiguous exact timing."""
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        for expected, line in enumerate(stream):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReplayError(
                    f"invalid annotation JSON at line {expected + 1}"
                ) from error
            index = value.get("frame_index", value.get("index"))
            timestamp = value.get("timestamp_ns")
            expected_timestamp = rational_timestamp_ns(
                expected, fps_numerator, fps_denominator
            )
            if index != expected or timestamp != expected_timestamp:
                raise ReplayError(f"annotation alignment error at frame {expected}")
            yield value


def _profile_sequence(profile: Any) -> int | None:
    """Return the production profile sequence or None."""
    if not isinstance(profile, dict):
        return None
    value = profile.get("frame_seq")
    return int(value) if isinstance(value, int) else None


def _copy_output(value: Any) -> Any:
    """Unwrap timing and recursively copy outputs into stable plain values."""
    raw = unwrap_timed(value)
    if isinstance(raw, np.ndarray):
        return raw.copy()
    if isinstance(raw, (list, tuple)):
        return [_copy_output(item) for item in raw]
    if isinstance(raw, dict):
        return {str(key): _copy_output(item) for key, item in raw.items()}
    if hasattr(raw, "tag_id") and hasattr(raw, "corners"):
        return {
            key: _copy_output(getattr(raw, key))
            for key in ("tag_id", "corners", "center", "decision_margin", "hamming")
            if hasattr(raw, key)
        }
    return raw


def collect_pipeline_outputs(pipeline: Any, frame_index: int) -> dict[str, Any]:
    """Snapshot detector, PnP, robot, timing, profile, and search regions."""
    outputs: dict[str, Any] = {"frame_index": frame_index}
    requested = (
        ("bench-detect", "detections", "detections"),
        ("bench-pnp", "camera_pose", "camera_pose"),
        ("bench-pnp", "pose_meta", "pose_meta"),
        ("bench-robot", "robot_pose", "robot_pose"),
    )
    for operation_uuid, port, key in requested:
        value = pipeline.get_operation_output(operation_uuid, port)
        timing = get_timing(value)
        if timing is not None and timing.frame_seq != frame_index:
            raise ReplayError(
                f"output {key} belongs to frame {timing.frame_seq}, not {frame_index}"
            )
        outputs[key] = _copy_output(value) if value is not None else None

    detector = pipeline.get_operation_by_uuid("bench-detect")
    if detector is not None:
        instance = detector.instance
        lock = getattr(instance, "last_detections_lock", None)
        if lock is not None:
            with lock:
                outputs["search_regions"] = [
                    np.asarray(region).copy().tolist()
                    for region in getattr(instance, "last_search_regions", [])
                ]

    camera_pose = outputs.get("camera_pose")
    if camera_pose is not None:
        raw_camera = np.asarray(camera_pose, dtype=float)
        outputs["camera_pose_raw_edn"] = raw_camera.copy()
    robot_pose = outputs.get("robot_pose")
    if robot_pose is not None:
        raw_robot = np.asarray(robot_pose, dtype=float)
        outputs["robot_pose_raw_edn"] = raw_robot.copy()
        outputs["robot_pose_nwu"] = pose_local_edn_to_nwu(raw_robot)
    outputs["profile"] = pipeline.get_latest_profile_snapshot()
    return outputs


def run_accuracy(
    decoder: Any,
    manager: ReplayCameraManager,
    pipeline: Any,
    fps_numerator: int,
    fps_denominator: int = 1,
    collect: Callable[[Any, int], T] | None = None,
    aligned_truth: Iterator[dict[str, Any]] | None = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    retain_records: bool = True,
) -> list[Any]:
    """Attempt every aligned frame and optionally stream completed records."""
    records: list[Any] = []
    truth = iter(aligned_truth) if aligned_truth is not None else None
    collector = collect or collect_pipeline_outputs
    expected = 0
    try:
        for item in decoder:
            if item.index != expected:
                raise ReplayError(
                    f"frame alignment error: expected {expected}, got {item.index}"
                )
            annotation = next(truth, None) if truth is not None else None
            if truth is not None and (
                annotation is None
                or annotation.get("frame_index", annotation.get("index")) != expected
            ):
                raise ReplayError(f"truth/video alignment error at frame {expected}")
            timestamp = rational_timestamp_ns(
                item.index, fps_numerator, fps_denominator
            )
            manager.publish(item.image, item.index, timestamp)
            manager.begin_cycle()
            before = _profile_sequence(pipeline.get_latest_profile_snapshot())
            started = time.perf_counter_ns()
            try:
                pipeline.run()
                duration = time.perf_counter_ns() - started
                errors = pipeline.get_operation_errors()
                if errors:
                    completed = False
                    value = None
                    failure = f"pipeline operation errors: {errors}"
                else:
                    value = collector(pipeline, item.index)
                    after = _profile_sequence(pipeline.get_latest_profile_snapshot())
                    completed = after is not None and after != before
                    failure = None
                    if isinstance(value, dict) and not completed:
                        value["profile"] = None
                record = {
                    "frame_index": expected,
                    "timestamp_ns": timestamp,
                    "attempted": True,
                    "completed": completed,
                    "skipped": not completed and failure is None,
                    "failure": failure,
                    "pipeline_duration_ns": duration,
                    "decode_duration_ns": item.decode_duration_ns,
                    "truth": annotation,
                    "output": value,
                }
            finally:
                manager.end_cycle()
            if on_record is not None:
                on_record(record)
            if retain_records:
                records.append(record if truth is not None else record["output"])
            expected += 1
        if truth is not None and next(truth, None) is not None:
            raise ReplayError("truth contains more frames than video")
        manager.mark_eof()
        return records
    except BaseException as error:
        manager.mark_error(error)
        raise


def _graph_edges(nodes: list[dict[str, Any]]) -> set[tuple[str, str, str, str, bool]]:
    """Normalize graph edges to operation names instead of UUIDs."""
    names = {
        str(node["uuid"]): str(node["action_name"]).removesuffix(".py")
        for node in nodes
    }
    relevant = {
        "device_input",
        "temporal_acceleration_preprocessor_rust",
        "detect_apriltags",
        "minimum_apriltag_count",
        "pnp_camera_localization",
        "camera_to_robot_pose",
    }
    return {
        (
            names[str(connection["from_uuid"])],
            str(connection["from_port"]),
            names[str(connection["to_uuid"])],
            str(connection["to_port"]),
            bool(connection.get("is_default", False)),
        )
        for node in nodes
        for connection in node.get("connections", [])
        if names.get(str(connection.get("from_uuid"))) in relevant
        and names.get(str(connection.get("to_uuid"))) in relevant
    }


def _validate_template_contract(
    data: list[dict[str, Any]], expected_temporal: bool
) -> None:
    """Fail visibly when production preset operation ports or feedback drift."""
    templates = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    name = "apriltag_localization" if expected_temporal else "basic_localization"
    template = templates[name]["nodes"]
    if _graph_edges(data) != _graph_edges(template):
        raise ReplayError(f"benchmark graph no longer matches {name} template ports")


def load_benchmark_config(
    path: str | Path, expected_temporal: bool
) -> list[dict[str, Any]]:
    """Load a matched graph and enforce required production operation identity."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ReplayError("benchmark pipeline configuration must be a JSON list")
    names = [str(node.get("action_name", "")).removesuffix(".py") for node in data]
    required = {
        "device_input",
        "detect_apriltags",
        "minimum_apriltag_count",
        "pnp_camera_localization",
        "camera_to_robot_pose",
    }
    if not required.issubset(names):
        raise ReplayError(
            f"benchmark graph lacks required operations: {sorted(required - set(names))}"
        )
    temporal = "temporal_acceleration_preprocessor_rust" in names
    if temporal != expected_temporal:
        raise ReplayError(
            "benchmark graph temporal identity does not match requested mode"
        )
    forbidden = {"publish_to_networktables", "camera_pose_output", "robot_pose_output"}
    if forbidden & set(names):
        raise ReplayError("benchmark graph contains a UI or publishing sink")
    if len(names) != len(set(names)):
        raise ReplayError("benchmark graph contains duplicate operation identities")
    _validate_template_contract(data, expected_temporal)
    return data
