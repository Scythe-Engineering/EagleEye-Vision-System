from __future__ import annotations

from struct import pack
from typing import Any, Mapping

from src.utils.flatpack_schema.base import FlatpackSchema


class Vector2Schema(FlatpackSchema):
    """Schema that serializes 2D vector data (position)."""

    schema_name = "vector2"

    def can_handle(self, value: Any) -> bool:
        """Return True when value contains 2D position data."""
        if not isinstance(value, Mapping):
            return False

        required_keys = {"x", "y"}
        if not all(key in value for key in required_keys):
            return False

        if "z" in value:
            return False

        try:
            float(value["x"])
            float(value["y"])
            return True
        except (TypeError, ValueError):
            return False

    def serialize(self, value: Any) -> bytes:
        """Serialize 2D vector data into Flatpack binary payload."""
        if not self.can_handle(value):
            raise ValueError(
                "Value must be a dict with x and y keys containing numeric values"
            )

        x = float(value["x"])
        y = float(value["y"])

        return pack("<ff", x, y)


class Vector3Schema(FlatpackSchema):
    """Schema that serializes 3D vector data (position)."""

    schema_name = "vector3"

    def can_handle(self, value: Any) -> bool:
        """Return True when value contains 3D position data."""
        if not isinstance(value, Mapping):
            return False

        required_keys = {"x", "y", "z"}
        if not all(key in value for key in required_keys):
            return False

        try:
            float(value["x"])
            float(value["y"])
            float(value["z"])
            return True
        except (TypeError, ValueError):
            return False

    def serialize(self, value: Any) -> bytes:
        """Serialize 3D vector data into Flatpack binary payload."""
        if not self.can_handle(value):
            raise ValueError(
                "Value must be a dict with x, y, and z keys containing numeric values"
            )

        x = float(value["x"])
        y = float(value["y"])
        z = float(value["z"])

        return pack("<fff", x, y, z)


class Pose2DSchema(FlatpackSchema):
    """Schema that serializes 2D pose data (position and rotation)."""

    schema_name = "pose2d"

    def can_handle(self, value: Any) -> bool:
        """Return True when value contains 2D position and rotation data."""
        if not isinstance(value, Mapping):
            return False

        required_keys = {"x", "y", "rotation"}
        if not all(key in value for key in required_keys):
            return False

        try:
            float(value["x"])
            float(value["y"])
            float(value["rotation"])
            return True
        except (TypeError, ValueError):
            return False

    def serialize(self, value: Any) -> bytes:
        """Serialize 2D pose data into Flatpack binary payload."""
        if not self.can_handle(value):
            raise ValueError(
                "Value must be a dict with x, y, and rotation keys containing numeric values"
            )

        x = float(value["x"])
        y = float(value["y"])
        rotation = float(value["rotation"])

        return pack("<fff", x, y, rotation)


class Pose3DSchema(FlatpackSchema):
    """Schema that serializes 3D pose data (position and rotation as Euler angles)."""

    schema_name = "pose3d"

    def can_handle(self, value: Any) -> bool:
        """Return True when value contains 3D position and rotation data."""
        if not isinstance(value, Mapping):
            return False

        required_keys = {"x", "y", "z", "roll", "pitch", "yaw"}
        if not all(key in value for key in required_keys):
            return False

        try:
            float(value["x"])
            float(value["y"])
            float(value["z"])
            float(value["roll"])
            float(value["pitch"])
            float(value["yaw"])
            return True
        except (TypeError, ValueError):
            return False

    def serialize(self, value: Any) -> bytes:
        """Serialize 3D pose data into Flatpack binary payload."""
        if not self.can_handle(value):
            raise ValueError(
                "Value must be a dict with x, y, z, roll, pitch, and yaw keys containing numeric values"
            )

        x = float(value["x"])
        y = float(value["y"])
        z = float(value["z"])
        roll = float(value["roll"])
        pitch = float(value["pitch"])
        yaw = float(value["yaw"])

        return pack("<ffffff", x, y, z, roll, pitch, yaw)
