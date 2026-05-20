"""Dummy dependency implementations used by tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from src.utils.timing import FramePacket, TimedValue, TimingMetadata


class DummyComputePool:
    """ComputePool stub to avoid importing torch-dependent modules."""

    def add_compute_device(self, compute_device: Any) -> None:
        """Add compute device.
        
        Args:
            compute_device (Any): Compute device."""
        return None

    def remove_compute_device(self, compute_device: Any) -> None:
        """Remove compute device.
        
        Args:
            compute_device (Any): Compute device."""
        return None

    def remove_compute_device_by_id(self, compute_device_id: str) -> None:
        """Remove compute device by id.
        
        Args:
            compute_device_id (str): Compute device id."""
        return None

    def get_compute_device(self, compute_device_id: str) -> Any:
        """Get compute device.
        
        Args:
            compute_device_id (str): Compute device id.
        
        Returns:
            Any: Result of get compute device."""
        raise KeyError(compute_device_id)

    def get_compute_devices_by_type(self, compute_device_type: str) -> list[Any]:
        """Get compute devices by type.
        
        Args:
            compute_device_type (str): Compute device type.
        
        Returns:
            list[Any]: Result of get compute devices by type."""
        return []

    def stop_all_devices(self) -> None:
        """Stop all devices."""
        return None


class FakeEagleEyeInterface:
    """No-op EagleEyeInterface replacement for tests."""

    def __init__(self) -> None:
        """Initialize the object."""
        self.detected_objects_payloads: list[list[dict[str, Any]]] = []
        self.robot_positions: list[np.ndarray] = []
        self.camera_poses: list[tuple[str, np.ndarray]] = []
        self.operation_errors: list[dict[str, Any]] = []
        self.camera_frames: list[tuple[str, np.ndarray]] = []

    def update_robot_position(self, transformation_matrix: np.ndarray) -> None:
        """Update robot position.
        
        Args:
            transformation_matrix (np.ndarray): Transformation matrix."""
        self.robot_positions.append(transformation_matrix)

    def update_detected_objects(self, detections: list[dict[str, Any]]) -> None:
        """Update detected objects.
        
        Args:
            detections (list[dict[str, Any]]): Detections."""
        self.detected_objects_payloads.append(detections)

    def update_camera_pose(
        self, camera_bus_id: str, transformation_matrix: np.ndarray
    ) -> None:
        """Update camera pose.
        
        Args:
            camera_bus_id (str): Camera bus id.
            transformation_matrix (np.ndarray): Transformation matrix."""
        self.camera_poses.append((camera_bus_id, transformation_matrix))

    def publish_operation_errors(self, payload: dict[str, Any]) -> None:
        """Publish operation errors.
        
        Args:
            payload (dict[str, Any]): Payload."""
        self.operation_errors.append(payload)

    def update_camera_frame(self, camera_name: str, frame: np.ndarray) -> None:
        """Update camera frame.
        
        Args:
            camera_name (str): Camera name.
            frame (np.ndarray): Frame."""
        self.camera_frames.append((camera_name, frame))


class FakeNetworkTable:
    """In-memory NetworkTables stub used by tests."""

    def __init__(self) -> None:
        """Initialize the object."""
        self.values: Dict[str, Any] = {}

    _NETWORKTABLES_API_ALIASES = {
        "getNumber": "get_number",
        "getEntry": "get_entry",
        "putRaw": "put_raw",
        "getStructTopic": "get_struct_topic",
        "getStructArrayTopic": "get_struct_array_topic",
        "getDoubleTopic": "get_double_topic",
        "getBooleanTopic": "get_boolean_topic",
        "getStringTopic": "get_string_topic",
        "getDoubleArrayTopic": "get_double_array_topic",
        "getBooleanArrayTopic": "get_boolean_array_topic",
        "getStringArrayTopic": "get_string_array_topic",
    }

    def __getattr__(self, name: str) -> Any:
        """Return WPILib-style API aliases without defining camelCase methods.
        
        Args:
            name (str): Attribute name.
        
        Returns:
            Any: Aliased snake_case method.
        """
        if name in self._NETWORKTABLES_API_ALIASES:
            return getattr(self, self._NETWORKTABLES_API_ALIASES[name])
        raise AttributeError(name)

    def get_number(self, key: str, default: Any) -> Any:
        """Get a number value.
        
        Args:
            key (str): Key.
            default (Any): Default value.
        
        Returns:
            Any: Stored value or default.
        """
        return self.values.get(key, default)

    def get_entry(self, key: str) -> Any:
        """Get a fake NetworkTables entry.
        
        Args:
            key (str): Key.
        
        Returns:
            Any: Fake entry object.
        """
        table = self

        class _Entry:
            def __getattr__(self, name: str) -> Any:
                """Return WPILib-style entry aliases.
                
                Args:
                    name (str): Attribute name.
                
                Returns:
                    Any: Aliased snake_case method.
                """
                if name == "getDouble":
                    return self.get_double
                raise AttributeError(name)

            def get_double(self, default: float) -> float:
                """Get a double value.
                
                Args:
                    default (float): Default value.
                
                Returns:
                    float: Stored numeric value or default.
                """
                value = table.values.get(key, default)
                return value if isinstance(value, float | int) else default

        return _Entry()

    def put_raw(self, key: str, value: bytes) -> None:
        """Put raw bytes.
        
        Args:
            key (str): Key.
            value (bytes): Value.
        """
        self.values[key] = value

    def get_struct_topic(self, key: str, _value_type: type) -> Any:
        """Get a struct topic.
        
        Args:
            key (str): Key.
            _value_type (type): Value type.
        
        Returns:
            Any: Fake topic.
        """
        return self._make_topic(key)

    def get_struct_array_topic(self, key: str, _value_type: type) -> Any:
        """Get a struct array topic.
        
        Args:
            key (str): Key.
            _value_type (type): Value type.
        
        Returns:
            Any: Fake topic.
        """
        return self._make_topic(key)

    def get_double_topic(self, key: str) -> Any:
        """Get a double topic.
        
        Args:
            key (str): Key.
        
        Returns:
            Any: Fake topic.
        """
        return self._make_topic(key)

    def get_boolean_topic(self, key: str) -> Any:
        """Get a boolean topic.
        
        Args:
            key (str): Key.
        
        Returns:
            Any: Fake topic.
        """
        return self._make_topic(key)

    def get_string_topic(self, key: str) -> Any:
        """Get a string topic.
        
        Args:
            key (str): Key.
        
        Returns:
            Any: Fake topic.
        """
        return self._make_topic(key)

    def get_double_array_topic(self, key: str) -> Any:
        """Get a double array topic.
        
        Args:
            key (str): Key.
        
        Returns:
            Any: Fake topic.
        """
        return self._make_topic(key)

    def get_boolean_array_topic(self, key: str) -> Any:
        """Get a boolean array topic.
        
        Args:
            key (str): Key.
        
        Returns:
            Any: Fake topic.
        """
        return self._make_topic(key)

    def get_string_array_topic(self, key: str) -> Any:
        """Get a string array topic.
        
        Args:
            key (str): Key.
        
        Returns:
            Any: Fake topic.
        """
        return self._make_topic(key)

    def _make_topic(self, key: str) -> Any:
        """Make topic.
        
        Args:
            key (str): Key.
        
        Returns:
            Any: Result of make topic."""
        table = self

        class _Topic:
            def publish(self) -> Any:
                """Publish.
                
                Returns:
                    Any: Result of publish."""
                class _Publisher:
                    def set(self, value: Any, timestamp: int | None = None) -> None:
                        """Set.
                        
                        Args:
                            value (Any): Value.
                            timestamp (int | None): Timestamp."""
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
    capture_nt_us: int = 1

    def get_current_packet(self) -> FramePacket | None:
        """Get current packet.
        
        Returns:
            FramePacket | None: Result of get current packet."""
        if self.frame is None:
            return None
        return TimedValue(
            self.frame,
            TimingMetadata(capture_nt_us=self.capture_nt_us),
        )


@dataclass
class FakeCameraThreadManager:
    """In-memory camera manager that returns dummy frames."""

    default_frame: np.ndarray
    camera_objects: Dict[str, FakeCameraWorker] = field(default_factory=dict)
    bus_id_to_name: Dict[str, str] = field(default_factory=dict)

    def add_camera(self, camera_name: str, frame: Optional[np.ndarray] = None) -> None:
        """Add camera.
        
        Args:
            camera_name (str): Camera name.
            frame (Optional[np.ndarray]): Frame."""
        self.camera_objects[camera_name] = FakeCameraWorker(
            camera_index=len(self.camera_objects),
            frame=frame if frame is not None else self.default_frame,
        )

    def register_bus_id(self, bus_id: str, camera_name: str) -> None:
        """Register bus id.
        
        Args:
            bus_id (str): Bus id.
            camera_name (str): Camera name."""
        self.bus_id_to_name[bus_id] = camera_name

    def get_camera_name_by_bus_id(self, bus_id: str) -> Optional[str]:
        """Get camera name by bus id.
        
        Args:
            bus_id (str): Bus id.
        
        Returns:
            Optional[str]: Result of get camera name by bus id."""
        return self.bus_id_to_name.get(bus_id)

    def get_current_packet(self, camera_name: str) -> FramePacket | None:
        """Get current packet.
        
        Args:
            camera_name (str): Camera name.
        
        Returns:
            FramePacket | None: Result of get current packet."""
        worker = self.camera_objects.get(camera_name)
        if worker is None:
            return None
        return worker.get_current_packet()

    def get_current_packet_by_bus_id(self, bus_id: str) -> FramePacket | None:
        """Get current packet by bus id.
        
        Args:
            bus_id (str): Bus id.
        
        Returns:
            FramePacket | None: Result of get current packet by bus id."""
        camera_name = self.get_camera_name_by_bus_id(bus_id)
        if camera_name is None:
            return None
        worker = self.camera_objects.get(camera_name)
        if worker is None:
            return None
        return worker.get_current_packet()

    def get_all_bus_ids(self) -> list[str]:
        """Get all bus ids.
        
        Returns:
            list[str]: Result of get all bus ids."""
        return list(self.bus_id_to_name.keys())
