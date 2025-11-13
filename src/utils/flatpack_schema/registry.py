from __future__ import annotations

from typing import Any, Iterable

from src.utils.flatpack_schema.array_schemas import (
    FloatArraySchema,
    Pose2DArraySchema,
    Pose3DArraySchema,
    Vector2ArraySchema,
    Vector3ArraySchema,
)
from src.utils.flatpack_schema.object_schemas import (
    Pose2DSchema,
    Pose3DSchema,
    Vector2Schema,
    Vector3Schema,
)
from src.utils.flatpack_schema.base import FlatpackSchema


class FlatpackRegistry:
    """Registry responsible for managing Flatpack schemas."""

    def __init__(self, schemas: Iterable[FlatpackSchema] | None = None) -> None:
        """Initialize the registry with an optional iterable of schemas."""
        self._schemas: list[FlatpackSchema] = []
        if schemas is not None:
            for schema in schemas:
                self.register(schema)

    def register(self, schema: FlatpackSchema) -> None:
        """Register a new Flatpack schema."""
        self._schemas.append(schema)

    def serialize(self, value: Any) -> tuple[bytes, str]:
        """Serialize the value using the first schema that can handle it."""
        for schema in self._schemas:
            if schema.can_handle(value):
                payload = schema.serialize(value)
                encoded_payload = _wrap_payload(schema.schema_name, payload)
                return encoded_payload, schema.schema_name
        raise ValueError(
            f"No Flatpack schema available for the provided value, value of type {type(value)}"
        )


def _wrap_payload(schema_name: str, payload: bytes) -> bytes:
    """Wrap a payload with header metadata including schema name."""
    schema_bytes = schema_name.encode("utf-8")
    if len(schema_bytes) > 255:
        raise ValueError("Schema name length exceeds Flatpack header limit")
    header = b"FPK1" + bytes([len(schema_bytes)]) + schema_bytes
    return header + payload


registry = FlatpackRegistry(
    schemas=[
        # Single objects (most specific, 3D before 2D to avoid false matches)
        Pose3DSchema(),
        Pose2DSchema(),
        Vector3Schema(),  # 3D before 2D to avoid false matches
        Vector2Schema(),
        # Arrays (ordered from most to least specific)
        Pose3DArraySchema(),
        Pose2DArraySchema(),
        Vector3ArraySchema(),  # 3D before 2D to avoid false matches
        Vector2ArraySchema(),
        FloatArraySchema(),
    ]
)
