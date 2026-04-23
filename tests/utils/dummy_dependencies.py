"""Dummy dependency implementations used by tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


class DummyComputePool:
    """ComputePool stub to avoid importing torch-dependent modules."""

    def add_compute_device(self, compute_device: Any) -> None:
        return None

    def remove_compute_device(self, compute_device: Any) -> None:
        return None

    def remove_compute_device_by_id(self, compute_device_id: str) -> None:
        return None

    def get_compute_device(self, compute_device_id: str) -> Any:
        raise KeyError(compute_device_id)

    def get_compute_devices_by_type(self, compute_device_type: str) -> list[Any]:
        return []

    def stop_all_devices(self) -> None:
        return None


class FakeEagleEyeInterface:
    """No-op EagleEyeInterface replacement for tests."""

    def __init__(self) -> None:
        self.detected_objects_payloads: list[list[dict[str, Any]]] = []
        self.robot_positions: list[np.ndarray] = []
        self.camera_poses: list[tuple[str, np.ndarray]] = []
        self.operation_errors: list[dict[str, Any]] = []
        self.camera_frames: list[tuple[str, np.ndarray]] = []

    def update_robot_position(self, transformation_matrix: np.ndarray) -> None:
        self.robot_positions.append(transformation_matrix)

    def update_detected_objects(self, detections: list[dict[str, Any]]) -> None:
        self.detected_objects_payloads.append(detections)

    def update_camera_pose(
        self, camera_bus_id: str, transformation_matrix: np.ndarray
    ) -> None:
        self.camera_poses.append((camera_bus_id, transformation_matrix))

    def publish_operation_errors(self, payload: dict[str, Any]) -> None:
        self.operation_errors.append(payload)

    def update_camera_frame(self, camera_name: str, frame: np.ndarray) -> None:
        self.camera_frames.append((camera_name, frame))


class FakeNetworkTable:
    """In-memory NetworkTables stub used by tests."""

    def __init__(self) -> None:
        self.values: Dict[str, Any] = {}

    def getNumber(self, key: str, default: Any) -> Any:
        return self.values.get(key, default)

    def getEntry(self, key: str) -> Any:  # noqa: N802
        table = self

        class _Entry:
            def getDouble(self, default: float) -> float:  # noqa: N802
                value = table.values.get(key, default)
                return value if isinstance(value, float | int) else default

        return _Entry()

    def putRaw(self, key: str, value: bytes) -> None:
        self.values[key] = value

    def getStructTopic(self, key: str, _value_type: type) -> Any:  # noqa: N802
        return self._make_topic(key)

    def getStructArrayTopic(self, key: str, _value_type: type) -> Any:  # noqa: N802
        return self._make_topic(key)

    def _make_topic(self, key: str) -> Any:
        table = self

        class _Topic:
            def publish(self) -> Any:
                class _Publisher:
                    def set(self, value: Any) -> None:
                        table.values[key] = value

                return _Publisher()

        return _Topic()


@dataclass
class FakeCameraWorker:
    """Minimal camera worker to emulate camera objects."""

    camera_index: int = 0
    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0

    def get_current_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        if self.frame is None:
            return None
        return self.frame, self.timestamp


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

    def get_current_frame(self, camera_name: str) -> Optional[Tuple[np.ndarray, float]]:
        """Get the latest frame tuple for a camera by name.

        Args:
            camera_name: Camera name to query.

        Returns:
            Optional[Tuple[np.ndarray, float]]: ``(frame, timestamp)`` when the
            camera exists and has a frame, otherwise ``None``.
        """
        worker = self.camera_objects.get(camera_name)
        if worker is None:
            return None
        return worker.get_current_frame()

    def get_current_frame_by_bus_id(
        self, bus_id: str
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Get the latest frame tuple for a camera by bus ID.

        Args:
            bus_id: Camera bus identifier.

        Returns:
            Optional[Tuple[np.ndarray, float]]: ``(frame, timestamp)`` when the
            bus ID maps to a camera with an available frame, otherwise ``None``.
        """
        camera_name = self.get_camera_name_by_bus_id(bus_id)
        if camera_name is None:
            return None
        return self.get_current_frame(camera_name)

    def get_all_bus_ids(self) -> list[str]:
        """Return all registered camera bus IDs.

        Args:
            None.

        Returns:
            list[str]: Registered bus IDs in insertion order.
        """
        return list(self.bus_id_to_name.keys())
