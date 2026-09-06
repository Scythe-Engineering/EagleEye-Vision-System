"""Dummy dependency implementations used by tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from tests.utils.dummy_data import dummy_frame
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.device_registry import DeviceNotFoundError
from src.utils.logging.logger import Logger
from src.utils.timing import FramePacket, TimedValue, TimingMetadata


class DummyDeviceRegistry:
    """Minimal startup device inventory used by pipeline tests."""

    def descriptors(self) -> tuple[Any, ...]:
        """Return the empty set of available device descriptors.

        Returns:
            tuple[Any, ...]: No device descriptors.
        """
        return ()

    def get(self, device_id: str) -> Any:
        """Reject lookup requests because this registry contains no devices.

        Args:
            device_id: Identifier of the requested device.

        Raises:
            DeviceNotFoundError: Always, because no devices are registered.
                Matches the real registry, which raises this KeyError subclass.
        """
        raise DeviceNotFoundError(device_id)


class DummyModelLibrary:
    """Marker model-library dependency for pipelines without detectors."""


class FakeEagleEyeInterface:
    """No-op EagleEyeInterface replacement for tests."""

    def __init__(self) -> None:
        self.detected_objects_payloads: list[list[dict[str, Any]]] = []
        self.robot_positions: list[np.ndarray] = []
        self.camera_poses: list[tuple[str, np.ndarray]] = []
        self.operation_errors: list[dict[str, Any]] = []
        self.camera_frames: list[tuple[str, np.ndarray]] = []

    def update_robot_position(self, transformation_matrix: np.ndarray) -> None:
        """Record a robot pose update.

        Args:
            transformation_matrix: Robot pose transformation matrix.
        """
        self.robot_positions.append(transformation_matrix)

    def update_detected_objects(self, detections: list[dict[str, Any]]) -> None:
        """Record a detected-object payload.

        Args:
            detections: Serialized detections to publish.
        """
        self.detected_objects_payloads.append(detections)

    def update_camera_pose(
        self, camera_bus_id: str, transformation_matrix: np.ndarray
    ) -> None:
        """Record a camera pose update.

        Args:
            camera_bus_id: Bus ID of the camera whose pose changed.
            transformation_matrix: Camera pose transformation matrix.
        """
        self.camera_poses.append((camera_bus_id, transformation_matrix))

    def publish_operation_errors(self, payload: dict[str, Any]) -> None:
        """Record an operation-error payload.

        Args:
            payload: Error data that would have been published.
        """
        self.operation_errors.append(payload)

    def update_camera_frame(self, camera_name: str, frame: np.ndarray) -> None:
        """Record a camera frame update.

        Args:
            camera_name: Name of the source camera.
            frame: Frame that would have been published.
        """
        self.camera_frames.append((camera_name, frame))


class FakeNetworkTable:
    """In-memory NetworkTables stub used by tests."""

    def __init__(self) -> None:
        self.values: Dict[str, Any] = {}

    def getNumber(self, key: str, default: Any) -> Any:
        """Return the stored numeric value or its default.

        Args:
            key: NetworkTables key to read.
            default: Value returned when the key is absent.

        Returns:
            Any: The stored value or ``default``.
        """
        return self.values.get(key, default)

    def getEntry(self, key: str) -> Any:  # noqa: N802
        """Return a lightweight entry for numeric reads.

        Args:
            key: NetworkTables key represented by the entry.

        Returns:
            Any: Entry stub with a ``getDouble`` method.
        """
        table = self

        class _Entry:
            def getDouble(self, default: float) -> float:  # noqa: N802
                """Return a stored numeric value or the provided default.

                Args:
                    default: Value returned for absent or nonnumeric entries.

                Returns:
                    float: Stored numeric value or ``default``.
                """
                value = table.values.get(key, default)
                return value if isinstance(value, float | int) else default

        return _Entry()

    def putRaw(self, key: str, value: bytes) -> None:
        """Store raw bytes under a NetworkTables key.

        Args:
            key: NetworkTables key to update.
            value: Raw payload to store.
        """
        self.values[key] = value

    def getStructTopic(self, key: str, _value_type: type) -> Any:  # noqa: N802
        """Return a publishing topic stub for a struct key.

        Args:
            key: NetworkTables key for the topic.
            _value_type: Ignored struct type.

        Returns:
            Any: Topic stub.
        """
        return self._make_topic(key)

    def getStructArrayTopic(self, key: str, _value_type: type) -> Any:  # noqa: N802
        """Return a publishing topic stub for a struct-array key.

        Args:
            key: NetworkTables key for the topic.
            _value_type: Ignored struct-array element type.

        Returns:
            Any: Topic stub.
        """
        return self._make_topic(key)

    def getDoubleTopic(self, key: str) -> Any:  # noqa: N802
        """Return a publishing topic stub for a double key.

        Args:
            key: NetworkTables key for the topic.

        Returns:
            Any: Topic stub.
        """
        return self._make_topic(key)

    def getBooleanTopic(self, key: str) -> Any:  # noqa: N802
        """Return a publishing topic stub for a boolean key.

        Args:
            key: NetworkTables key for the topic.

        Returns:
            Any: Topic stub.
        """
        return self._make_topic(key)

    def getStringTopic(self, key: str) -> Any:  # noqa: N802
        """Return a publishing topic stub for a string key.

        Args:
            key: NetworkTables key for the topic.

        Returns:
            Any: Topic stub.
        """
        return self._make_topic(key)

    def getDoubleArrayTopic(self, key: str) -> Any:  # noqa: N802
        """Return a publishing topic stub for a double-array key.

        Args:
            key: NetworkTables key for the topic.

        Returns:
            Any: Topic stub.
        """
        return self._make_topic(key)

    def getBooleanArrayTopic(self, key: str) -> Any:  # noqa: N802
        """Return a publishing topic stub for a boolean-array key.

        Args:
            key: NetworkTables key for the topic.

        Returns:
            Any: Topic stub.
        """
        return self._make_topic(key)

    def getStringArrayTopic(self, key: str) -> Any:  # noqa: N802
        """Return a publishing topic stub for a string-array key.

        Args:
            key: NetworkTables key for the topic.

        Returns:
            Any: Topic stub.
        """
        return self._make_topic(key)

    def _make_topic(self, key: str) -> Any:
        """Create a topic stub that persists published values.

        Args:
            key: NetworkTables key associated with the topic.

        Returns:
            Any: Topic stub.
        """
        table = self

        class _Topic:
            def publish(self, *_options: object, **_kwargs: object) -> Any:
                """Create a publisher for this topic.

                Returns:
                    Any: Publisher stub.
                """

                class _Publisher:
                    def set(self, value: Any, timestamp: int | None = None) -> None:
                        """Persist a published value and optional timestamp.

                        Args:
                            value: Value to publish.
                            timestamp: Optional NetworkTables timestamp.
                        """
                        table.values[key] = value
                        if timestamp is not None:
                            table.values[f"{key}:timestamp"] = timestamp

                return _Publisher()

        return _Topic()


@dataclass
class FakeCameraWorker:
    """Minimal camera worker to emulate camera objects."""

    camera_index: int = 0
    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    running: bool = True
    thread: Optional[object] = None
    frame_seq: int = 0

    def _current_timing(self) -> Optional[TimingMetadata]:
        """Build timing for the fake's current frame without advancing it."""
        if self.frame is None:
            return None
        return TimingMetadata(
            capture_nt_us=int(self.timestamp * 1000) or 1,
            capture_monotonic_ns=1,
            camera_name="fake_camera",
            frame_seq=self.frame_seq,
        )

    def get_current_packet(self) -> FramePacket | None:
        """Return the latest frame with deterministic timing metadata.

        Each call advances ``frame_seq`` so consumers that poll for a frame
        newer than the one they last consumed observe a new packet.

        Returns:
            FramePacket | None: Current packet, or ``None`` without a frame.
        """
        if self.frame is None:
            return None
        self.frame_seq += 1
        timing = self._current_timing()
        assert timing is not None
        return TimedValue(self.frame, timing)

    def get_current_timing(self) -> Optional[TimingMetadata]:
        """Return the current packet's timing without advancing its sequence."""
        return self._current_timing()


@dataclass
class FakeCameraThreadManager:
    """In-memory camera manager that returns dummy frames."""

    default_frame: np.ndarray
    camera_objects: Dict[str, FakeCameraWorker] = field(default_factory=dict)
    bus_id_to_name: Dict[str, str] = field(default_factory=dict)

    def add_camera(self, camera_name: str, frame: Optional[np.ndarray] = None) -> None:
        """Add a fake camera worker to the in-memory registry.

        Args:
            camera_name: Human-readable camera name.
            frame: Optional frame override for this camera. If omitted,
                ``default_frame`` is used.

        Returns:
            None.
        """
        self.camera_objects[camera_name] = FakeCameraWorker(
            camera_index=len(self.camera_objects),
            frame=frame if frame is not None else self.default_frame,
        )

    @property
    def cameras(self) -> Dict[str, FakeCameraWorker]:
        """Expose workers under the name used by ``CameraThreadManager``.

        Returns:
            Dict[str, FakeCameraWorker]: Registered workers keyed by name.
        """
        return self.camera_objects

    def wait_for_new_frame_by_bus_id(
        self,
        bus_id: str,
        after_frame_seq: int,
        timeout_s: Optional[float] = None,
    ) -> bool:
        """Report whether a newer frame exists for a camera bus ID.

        Args:
            bus_id: Camera bus identifier.
            after_frame_seq: Last frame sequence consumed by the caller.
            timeout_s: Accepted for interface parity; the fake never blocks.

        Returns:
            bool: ``True`` when a newer packet is available.
        """
        packet = self.get_current_packet_by_bus_id(bus_id)
        if packet is None or packet.timing.frame_seq is None:
            return False
        return packet.timing.frame_seq > after_frame_seq

    def register_bus_id(self, bus_id: str, camera_name: str) -> None:
        """Associate a deterministic bus ID with a camera name.

        Args:
            bus_id: Camera bus identifier.
            camera_name: Camera name mapped to the provided bus ID.

        Returns:
            None.
        """
        self.bus_id_to_name[bus_id] = camera_name

    def get_camera_name_by_bus_id(self, bus_id: str) -> Optional[str]:
        """Resolve a camera name from its bus ID.

        Args:
            bus_id: Camera bus identifier.

        Returns:
            Optional[str]: The mapped camera name, or ``None`` when missing.
        """
        return self.bus_id_to_name.get(bus_id)

    def get_current_timing_by_bus_id(self, bus_id: str) -> Optional[TimingMetadata]:
        """Get the latest capture timing for a camera bus ID.

        Args:
            bus_id: Camera bus identifier.

        Returns:
            Optional[TimingMetadata]: Current timing, or ``None`` when unavailable.
        """
        camera_name = self.get_camera_name_by_bus_id(bus_id)
        if camera_name is None:
            return None
        worker = self.camera_objects.get(camera_name)
        return worker.get_current_timing() if worker is not None else None

    def get_current_packet_by_bus_id(self, bus_id: str) -> FramePacket | None:
        """Get the latest timestamped frame packet for a camera bus ID.

        Args:
            bus_id: Camera bus identifier.

        Returns:
            FramePacket | None: Current packet, or ``None`` when unavailable.
        """
        camera_name = self.get_camera_name_by_bus_id(bus_id)
        if camera_name is None:
            return None
        worker = self.camera_objects.get(camera_name)
        if worker is None:
            return None
        return worker.get_current_packet()

    def get_all_bus_ids(self) -> list[str]:
        """Return all registered camera bus IDs.

        Returns:
            list[str]: Registered bus IDs in insertion order.
        """
        return list(self.bus_id_to_name.keys())


def build_dummy_dependencies() -> dict[str, Any]:
    """Build common non-hardware dependencies for operation smoke tests.

    Returns:
        dict[str, Any]: Constructor dependencies keyed by parameter name.
    """
    return {
        "web_interface": FakeEagleEyeInterface(),
        "device_registry": DummyDeviceRegistry(),
        "model_library": DummyModelLibrary(),
        "network_table": FakeNetworkTable(),
        "camera_manager": FakeCameraThreadManager(default_frame=dummy_frame()),
        "camera_config_registry": CameraConfigRegistry(),
        "logger": Logger(log_directory="logs/test"),
    }
