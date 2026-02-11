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
        self.operation_errors: list[dict[str, Any]] = []
        self.camera_frames: list[tuple[str, np.ndarray]] = []

    def update_robot_position(self, transformation_matrix: np.ndarray) -> None:
        self.robot_positions.append(transformation_matrix)

    def update_detected_objects(self, detections: list[dict[str, Any]]) -> None:
        self.detected_objects_payloads.append(detections)

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

    def putRaw(self, key: str, value: bytes) -> None:
        self.values[key] = value


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
        self.camera_objects[camera_name] = FakeCameraWorker(
            camera_index=len(self.camera_objects),
            frame=frame if frame is not None else self.default_frame,
        )

    def register_bus_id(self, bus_id: str, camera_name: str) -> None:
        self.bus_id_to_name[bus_id] = camera_name

    def get_camera_name_by_bus_id(self, bus_id: str) -> Optional[str]:
        return self.bus_id_to_name.get(bus_id)

    def get_current_frame(self, camera_name: str) -> Optional[Tuple[np.ndarray, float]]:
        worker = self.camera_objects.get(camera_name)
        if worker is None:
            return None
        return worker.get_current_frame()

    def get_current_frame_by_bus_id(
        self, bus_id: str
    ) -> Optional[Tuple[np.ndarray, float]]:
        camera_name = self.get_camera_name_by_bus_id(bus_id)
        if camera_name is None:
            return None
        return self.get_current_frame(camera_name)

    def get_all_bus_ids(self) -> list[str]:
        return list(self.bus_id_to_name.keys())
