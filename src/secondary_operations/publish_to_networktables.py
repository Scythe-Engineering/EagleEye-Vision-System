from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from wpimath.geometry import (
    Pose2d,
    Pose3d,
    Quaternion,
    Rotation2d,
    Rotation3d,
    Transform2d,
    Transform3d,
    Translation2d,
    Translation3d,
)

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.timing import get_timing, retime, unwrap_timed

_POSE3D_KEYS = frozenset({"x", "y", "z", "roll", "pitch", "yaw"})
_POSE2D_KEYS = frozenset({"x", "y", "rotation"})
_TRANSLATION3D_KEYS = frozenset({"x", "y", "z"})
_TRANSLATION2D_KEYS = frozenset({"x", "y"})
_EDN_TO_NWU_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)


def _matrix_to_pose3d(matrix: np.ndarray) -> Pose3d:
    x, y, z = float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])
    R = matrix[:3, :3] @ _EDN_TO_NWU_ROTATION
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, qx = 0.25 / s, (R[2, 1] - R[1, 2]) * s
        qy, qz = (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, qx = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        qy, qz = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, qx = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        qy, qz = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, qx = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        qy, qz = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return Pose3d(
        Translation3d(x, y, z),
        Rotation3d(Quaternion(float(w), float(qx), float(qy), float(qz))),
    )


def _matrix_to_pose2d(matrix: np.ndarray) -> Pose2d:
    x, y = float(matrix[0, 3]), float(matrix[1, 3])
    yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return Pose2d(Translation2d(x, y), Rotation2d(yaw))


def _dict_to_wpilib(value: dict, schema: str = "auto") -> Any:
    keys = frozenset(value.keys())
    if schema == "rotation3d" and {"roll", "pitch", "yaw"} <= keys:
        return Rotation3d(float(value["roll"]), float(value["pitch"]), float(value["yaw"]))
    if schema == "rotation2d" and "rotation" in keys:
        return Rotation2d(float(value["rotation"]))
    if schema == "transform3d" and _POSE3D_KEYS <= keys:
        return Transform3d(
            Translation3d(float(value["x"]), float(value["y"]), float(value["z"])),
            Rotation3d(float(value["roll"]), float(value["pitch"]), float(value["yaw"])),
        )
    if schema == "transform2d" and _POSE2D_KEYS <= keys and "z" not in keys:
        return Transform2d(
            Translation2d(float(value["x"]), float(value["y"])),
            Rotation2d(float(value["rotation"])),
        )
    if schema == "translation3d" and _TRANSLATION3D_KEYS <= keys:
        return Translation3d(float(value["x"]), float(value["y"]), float(value["z"]))
    if schema == "translation2d" and _TRANSLATION2D_KEYS <= keys and "z" not in keys:
        return Translation2d(float(value["x"]), float(value["y"]))
    if _POSE3D_KEYS <= keys:
        return Pose3d(
            Translation3d(float(value["x"]), float(value["y"]), float(value["z"])),
            Rotation3d(float(value["roll"]), float(value["pitch"]), float(value["yaw"])),
        )
    if _POSE2D_KEYS <= keys and "z" not in keys:
        return Pose2d(
            Translation2d(float(value["x"]), float(value["y"])),
            Rotation2d(float(value["rotation"])),
        )
    if _TRANSLATION3D_KEYS <= keys and "rotation" not in keys and "roll" not in keys:
        return Translation3d(float(value["x"]), float(value["y"]), float(value["z"]))
    if _TRANSLATION2D_KEYS <= keys and "z" not in keys and "rotation" not in keys:
        return Translation2d(float(value["x"]), float(value["y"]))
    return None


def _coerce_wpilib(value: Any, schema: str) -> Any:
    if schema in {"double", "float", "number"} and isinstance(value, int | float):
        return float(value)
    if schema in {"boolean", "bool"} and isinstance(value, bool):
        return value
    if schema == "string" and isinstance(value, str):
        return value
    if schema == "auto" and isinstance(value, (bool, int, float, str)):
        return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else value
    if isinstance(value, np.ndarray):
        if value.shape == (4, 4):
            return _matrix_to_pose2d(value) if schema == "pose2d" else _matrix_to_pose3d(value)
        if schema in {"double_array", "float_array", "number_array", "auto"} and np.issubdtype(value.dtype, np.number):
            return [float(item) for item in value.flatten().tolist()]
    if isinstance(value, dict):
        return _dict_to_wpilib(value, schema)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return []
        if all(isinstance(item, bool) for item in value):
            return list(value)
        if all(isinstance(item, int | float) and not isinstance(item, bool) for item in value):
            return [float(item) for item in value]
        if all(isinstance(item, str) for item in value):
            return list(value)
        items = [_coerce_wpilib(item, schema) for item in value]
        if items and all(item is not None for item in items) and all(type(i) is type(items[0]) for i in items):
            return items
    return None


class PublishToNetworktables(OperationInstance):
    def __init__(
        self,
        network_table: Any,
        target_key: str,
        schema: str = "auto",
        data_path: str | Sequence[str] | None = None,
    ) -> None:
        self.network_table = network_table
        self.target_key = target_key
        self.schema = schema
        self.data_path_tokens = self._normalize_path(data_path)
        self._publisher: Any = None
        self.uses_timed_inputs = True

    def run(self, data: Any) -> Any:
        value = self._select_value(data)
        if value is not None:
            self._publish(value)
        return data

    def update_config(self, json_config: dict) -> None:
        if "target_key" in json_config:
            self.target_key = json_config["target_key"]
            self._publisher = None
        if "schema" in json_config:
            self.schema = json_config["schema"]
        if "data_path" in json_config:
            self.data_path_tokens = self._normalize_path(json_config["data_path"])

    def _publish(self, value: Any) -> None:
        timing = get_timing(value)
        raw_value = unwrap_timed(value)
        wpi_value = _coerce_wpilib(raw_value, self.schema)
        if wpi_value is None:
            return
        if self._publisher is None:
            self._publisher = self._create_publisher(wpi_value)
            if self._publisher is None:
                return
        if timing is not None:
            self._publisher.set(wpi_value, timing.capture_nt_us)
        else:
            self._publisher.set(wpi_value)

    def _create_publisher(self, wpi_value: Any) -> Any:
        if isinstance(wpi_value, list):
            if not wpi_value:
                return None
            first = wpi_value[0]
            if isinstance(first, bool):
                return self.network_table.getBooleanArrayTopic(self.target_key).publish()
            if isinstance(first, float):
                return self.network_table.getDoubleArrayTopic(self.target_key).publish()
            if isinstance(first, str):
                return self.network_table.getStringArrayTopic(self.target_key).publish()
            return self.network_table.getStructArrayTopic(
                self.target_key, type(first)
            ).publish()
        if isinstance(wpi_value, bool):
            return self.network_table.getBooleanTopic(self.target_key).publish()
        if isinstance(wpi_value, float):
            return self.network_table.getDoubleTopic(self.target_key).publish()
        if isinstance(wpi_value, str):
            return self.network_table.getStringTopic(self.target_key).publish()
        return self.network_table.getStructTopic(self.target_key, type(wpi_value)).publish()

    def _normalize_path(self, data_path: str | Sequence[str] | None) -> list[str | int]:
        if data_path is None:
            return []
        if isinstance(data_path, str):
            raw_tokens = [token for token in data_path.split(".") if token]
        else:
            raw_tokens = list(data_path)
        normalized: list[str | int] = []
        for token in raw_tokens:
            if isinstance(token, int):
                normalized.append(token)
            else:
                token_str = str(token).strip()
                normalized.append(int(token_str) if token_str.isdigit() else token_str)
        return normalized

    def _select_value(self, data: Any) -> Any:
        if not self.data_path_tokens:
            return data
        timing = get_timing(data)
        raw_data = unwrap_timed(data)
        if self._should_extract_sequence_field(raw_data):
            selected = self._extract_sequence_field(raw_data)
            return retime(selected, timing) if timing is not None and selected is not None else selected
        current = raw_data
        for token in self.data_path_tokens:
            if isinstance(token, int):
                if isinstance(current, Sequence):
                    try:
                        current = current[token]
                    except (IndexError, TypeError):
                        return None
                else:
                    return None
            else:
                if isinstance(current, dict) and token in current:
                    current = current[token]
                else:
                    return None
        return retime(current, timing) if timing is not None else current

    def _should_extract_sequence_field(self, data: Any) -> bool:
        return (
            len(self.data_path_tokens) == 1
            and isinstance(self.data_path_tokens[0], str)
            and isinstance(data, Sequence)
            and not isinstance(data, (str, bytes, bytearray))
        )

    def _extract_sequence_field(self, data: Sequence[Any]) -> Any:
        field_name = self.data_path_tokens[0]
        try:
            return [
                item[field_name]
                for item in data
                if isinstance(item, dict) and field_name in item
            ]
        except (KeyError, TypeError):
            return None
