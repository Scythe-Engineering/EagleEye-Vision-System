from __future__ import annotations

from numbers import Real
from struct import pack
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils.flatpack_schema.base import FlatpackSchema


def _is_collection(value: Any) -> bool:
    """Return True when the value behaves like a collection of items."""
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _coerce_number_sequence(value: Any) -> list[float] | None:
    """Convert a sequence of numbers into floats."""
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and np.issubdtype(value.dtype, np.floating):
            return value.tolist()
        elif value.ndim == 1 and np.issubdtype(value.dtype, np.number):
            return value.astype(float).tolist()
        return None
    if not _is_collection(value):
        return None
    float_values: list[float] = []
    for element in value:
        if isinstance(element, Real):
            float_values.append(float(element))
        else:
            return None
    return float_values


def _coerce_vector_sequence(
    value: Any, expected_axes: Sequence[str]
) -> list[list[float]] | None:
    """Convert a sequence of vector-like values into lists of floats."""
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[1] == len(expected_axes):
            if np.issubdtype(value.dtype, np.floating):
                return value.tolist()
            elif np.issubdtype(value.dtype, np.number):
                return value.astype(float).tolist()
        return None
    if not _is_collection(value):
        return None
    normalized_vectors: list[list[float]] = []
    for vector_candidate in value:
        vector_components = _extract_vector_components(vector_candidate, expected_axes)
        if vector_components is None:
            return None
        normalized_vectors.append(vector_components)
    return normalized_vectors


def _extract_vector_components(
    candidate: Any, expected_axes: Sequence[str]
) -> list[float] | None:
    """Convert a vector-like candidate into a list of floats when possible."""
    if isinstance(candidate, Mapping):
        try:
            return [float(candidate[axis]) for axis in expected_axes]
        except (KeyError, TypeError, ValueError):
            return None
    if _is_collection(candidate) and len(candidate) == len(expected_axes):
        vector_values: list[float] = []
        for element in candidate:
            if isinstance(element, Real):
                vector_values.append(float(element))
            else:
                return None
        return vector_values
    return None


class FloatArraySchema(FlatpackSchema):
    """Schema that serializes sequences of floats."""

    schema_name = "float_array"

    def can_handle(self, value: Any) -> bool:
        """Return True when value is a sequence of float-like numbers."""
        return _coerce_number_sequence(value) is not None

    def serialize(self, value: Any) -> bytes:
        """Serialize a sequence of floats into Flatpack binary payload."""
        float_values = _coerce_number_sequence(value)
        if float_values is None:
            raise ValueError("Value is not a sequence of numbers")
        element_count = len(float_values)
        header = pack("<I", element_count)
        if element_count == 0:
            return header
        body = pack(f"<{element_count}f", *float_values)
        return header + body


class Vector2ArraySchema(FlatpackSchema):
    """Schema that serializes sequences of 2D vectors."""

    schema_name = "vector2_array"
    axes = ("x", "y")

    def can_handle(self, value: Any) -> bool:
        """Return True when value is a sequence of 2D vector-like entries."""
        return _coerce_vector_sequence(value, self.axes) is not None

    def serialize(self, value: Any) -> bytes:
        """Serialize a sequence of 2D vectors into Flatpack binary payload."""
        normalized_vectors = _coerce_vector_sequence(value, self.axes)
        if normalized_vectors is None:
            raise ValueError("Value is not a sequence of 2D vectors")
        vector_count = len(normalized_vectors)
        header = pack("<I", vector_count)
        if vector_count == 0:
            return header
        flattened: list[float] = [
            component for vector in normalized_vectors for component in vector
        ]
        body = pack(f"<{vector_count * 2}f", *flattened)
        return header + body


class Vector3ArraySchema(FlatpackSchema):
    """Schema that serializes sequences of 3D vectors."""

    schema_name = "vector3_array"
    axes = ("x", "y", "z")

    def can_handle(self, value: Any) -> bool:
        """Return True when value is a sequence of 3D vector-like entries."""
        return _coerce_vector_sequence(value, self.axes) is not None

    def serialize(self, value: Any) -> bytes:
        """Serialize a sequence of 3D vectors into Flatpack binary payload."""
        normalized_vectors = _coerce_vector_sequence(value, self.axes)
        if normalized_vectors is None:
            raise ValueError("Value is not a sequence of 3D vectors")
        vector_count = len(normalized_vectors)
        header = pack("<I", vector_count)
        if vector_count == 0:
            return header
        flattened: list[float] = [
            component for vector in normalized_vectors for component in vector
        ]
        body = pack(f"<{vector_count * 3}f", *flattened)
        return header + body


def _extract_pose2d_components(candidate: Any) -> list[float] | None:
    """Convert a 2D pose candidate into a list of floats when possible."""
    if isinstance(candidate, Mapping):
        required_keys = {"x", "y", "rotation"}
        if not all(key in candidate for key in required_keys):
            return None
        try:
            return [
                float(candidate["x"]),
                float(candidate["y"]),
                float(candidate["rotation"]),
            ]
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _extract_pose3d_components(candidate: Any) -> list[float] | None:
    """Convert a 3D pose candidate into a list of floats when possible."""
    if isinstance(candidate, Mapping):
        required_keys = {"x", "y", "z", "roll", "pitch", "yaw"}
        if not all(key in candidate for key in required_keys):
            return None
        try:
            return [
                float(candidate["x"]),
                float(candidate["y"]),
                float(candidate["z"]),
                float(candidate["roll"]),
                float(candidate["pitch"]),
                float(candidate["yaw"]),
            ]
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _coerce_pose2d_sequence(value: Any) -> list[list[float]] | None:
    """Convert a sequence of 2D pose-like values into lists of floats."""
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[1] == 3:
            if np.issubdtype(value.dtype, np.floating):
                return value.tolist()
            elif np.issubdtype(value.dtype, np.number):
                return value.astype(float).tolist()
        return None
    if not _is_collection(value):
        return None
    normalized_poses: list[list[float]] = []
    for pose_candidate in value:
        pose_components = _extract_pose2d_components(pose_candidate)
        if pose_components is None:
            return None
        normalized_poses.append(pose_components)
    return normalized_poses


def _coerce_pose3d_sequence(value: Any) -> list[list[float]] | None:
    """Convert a sequence of 3D pose-like values into lists of floats."""
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[1] == 6:
            if np.issubdtype(value.dtype, np.floating):
                return value.tolist()
            elif np.issubdtype(value.dtype, np.number):
                return value.astype(float).tolist()
        return None
    if not _is_collection(value):
        return None
    normalized_poses: list[list[float]] = []
    for pose_candidate in value:
        pose_components = _extract_pose3d_components(pose_candidate)
        if pose_components is None:
            return None
        normalized_poses.append(pose_components)
    return normalized_poses


class Pose2DArraySchema(FlatpackSchema):
    """Schema that serializes sequences of 2D poses."""

    schema_name = "pose2d_array"

    def can_handle(self, value: Any) -> bool:
        """Return True when value is a sequence of 2D pose-like entries."""
        return _coerce_pose2d_sequence(value) is not None

    def serialize(self, value: Any) -> bytes:
        """Serialize a sequence of 2D poses into Flatpack binary payload."""
        normalized_poses = _coerce_pose2d_sequence(value)
        if normalized_poses is None:
            raise ValueError("Value is not a sequence of 2D poses")
        pose_count = len(normalized_poses)
        header = pack("<I", pose_count)
        if pose_count == 0:
            return header
        flattened: list[float] = [
            component for pose in normalized_poses for component in pose
        ]
        body = pack(f"<{pose_count * 3}f", *flattened)
        return header + body


class Pose3DArraySchema(FlatpackSchema):
    """Schema that serializes sequences of 3D poses."""

    schema_name = "pose3d_array"

    def can_handle(self, value: Any) -> bool:
        """Return True when value is a sequence of 3D pose-like entries."""
        return _coerce_pose3d_sequence(value) is not None

    def serialize(self, value: Any) -> bytes:
        """Serialize a sequence of 3D poses into Flatpack binary payload."""
        normalized_poses = _coerce_pose3d_sequence(value)
        if normalized_poses is None:
            raise ValueError("Value is not a sequence of 3D poses")
        pose_count = len(normalized_poses)
        header = pack("<I", pose_count)
        if pose_count == 0:
            return header
        flattened: list[float] = [
            component for pose in normalized_poses for component in pose
        ]
        body = pack(f"<{pose_count * 6}f", *flattened)
        return header + body
