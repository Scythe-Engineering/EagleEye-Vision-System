# Flatpack Schema Creation Guide

This guide explains how to create new schemas for the Flatpack serialization system used by the `PublishToNetworktables` operation.

## Overview

Flatpack schemas define how to serialize Python data structures into efficient binary formats for NetworkTables. Each schema handles a specific data type or structure, converting it to a compact byte representation that can be quickly deserialized by NetworkTables consumers.

## Schema Architecture

Schemas are organized into two categories:

- **Array schemas** (`array_schemas.py`): Handle sequences/collections of data
- **Object schemas** (`object_schemas.py`): Handle single complex objects

### Base Class

All schemas inherit from `FlatpackSchema`:

```python
from src.utils.flatpack_schema.base import FlatpackSchema

class MyCustomSchema(FlatpackSchema):
    schema_name = "my_custom_schema"

    def can_handle(self, value: Any) -> bool:
        # Return True if this schema can serialize the value
        pass

    def serialize(self, value: Any) -> bytes:
        # Convert value to bytes
        pass
```

### Schema Registration

After creating a schema, register it with the global registry:

```python
from src.utils.flatpack_schema.registry import registry

registry.register(MyCustomSchema())
```

## Creating a Schema: Step-by-Step

### Step 1: Define the Data Structure

Choose what data structure your schema will handle. Common patterns:

- **Arrays of primitives**: `[1, 2, 3]`, `["a", "b", "c"]`
- **Arrays of objects**: `[{"x": 1, "y": 2}, {"x": 3, "y": 4}]`
- **Single complex objects**: `{"name": "value", "count": 42}`
- **Matrices or nested arrays**: `[[1, 2], [3, 4]]`

### Step 2: Implement `can_handle()`

This method determines if your schema can serialize a given value. It should:

- Check the overall structure (list, dict, etc.)
- Validate element types
- Return `False` for incompatible data
- Be fast (called during serialization)

Example patterns:

```python
def can_handle(self, value: Any) -> bool:
    # For arrays of strings
    if not isinstance(value, (list, tuple)):
        return False
    return all(isinstance(item, str) for item in value)

def can_handle(self, value: Any) -> bool:
    # For objects with specific keys
    if not isinstance(value, dict):
        return False
    return "required_key" in value and isinstance(value["required_key"], int)
```

### Step 3: Implement `serialize()`

Convert the value to bytes using Python's `struct` module for binary packing:

```python
from struct import pack

def serialize(self, value: Any) -> bytes:
    # Validate and normalize the data
    normalized = self._normalize_data(value)

    # Pack into binary format
    # Use little-endian (<) for cross-platform compatibility
    return pack("<format_string", *normalized)
```

Common `struct` format characters:
- `I`: unsigned int (4 bytes)
- `f`: float (4 bytes)
- `d`: double (8 bytes)
- `B`: unsigned char (1 byte)
- `H`: unsigned short (2 bytes)

### Step 4: Handle Edge Cases

Consider these scenarios:

- **Empty collections**: Handle `[]`, `{}`, `""` gracefully
- **Type coercion**: Convert ints to floats, handle numeric precision
- **Validation failures**: Raise `ValueError` for invalid data
- **Memory efficiency**: Avoid creating unnecessary intermediate objects

## Example Schemas

### String Array Schema

```python
from struct import pack
from typing import Any

class StringArraySchema(FlatpackSchema):
    """Schema for arrays of strings."""

    schema_name = "string_array"

    def can_handle(self, value: Any) -> bool:
        """Check if value is a sequence of strings."""
        if not isinstance(value, (list, tuple)):
            return False
        return all(isinstance(item, str) for item in value)

    def serialize(self, value: Any) -> bytes:
        """Serialize string array to bytes."""
        if not self.can_handle(value):
            raise ValueError("Value must be a sequence of strings")

        # Pack string count
        count = len(value)
        result = pack("<I", count)

        # Pack each string: length + UTF-8 bytes
        for string in value:
            string_bytes = string.encode("utf-8")
            result += pack("<I", len(string_bytes))
            result += string_bytes

        return result
```

### Boolean Array Schema

```python
from struct import pack
from typing import Any

class BooleanArraySchema(FlatpackSchema):
    """Schema for arrays of booleans."""

    schema_name = "boolean_array"

    def can_handle(self, value: Any) -> bool:
        """Check if value is a sequence of booleans."""
        if not isinstance(value, (list, tuple)):
            return False
        return all(isinstance(item, bool) for item in value)

    def serialize(self, value: Any) -> bytes:
        """Pack booleans as bits in bytes."""
        if not self.can_handle(value):
            raise ValueError("Value must be a sequence of booleans")

        count = len(value)
        result = pack("<I", count)  # Pack count

        # Pack 8 booleans per byte
        for i in range(0, count, 8):
            byte_value = 0
            for j in range(8):
                if i + j < count and value[i + j]:
                    byte_value |= (1 << j)
            result += pack("<B", byte_value)

        return result
```

### Matrix Schema (2D Float Array)

```python
from struct import pack
from typing import Any, Sequence

class MatrixSchema(FlatpackSchema):
    """Schema for 2D matrices of floats."""

    schema_name = "float_matrix"

    def can_handle(self, value: Any) -> bool:
        """Check if value is a 2D array of floats."""
        if not isinstance(value, (list, tuple)):
            return False
        if not value:
            return True  # Empty matrix is valid

        # Check if it's a sequence of sequences
        for row in value:
            if not isinstance(row, (list, tuple)):
                return False
            if not all(isinstance(cell, (int, float)) for cell in row):
                return False

        return True

    def serialize(self, value: Any) -> bytes:
        """Serialize 2D matrix to bytes."""
        if not self.can_handle(value):
            raise ValueError("Value must be a 2D array of numbers")

        rows = len(value)
        cols = len(value[0]) if value else 0

        # Pack dimensions
        result = pack("<II", rows, cols)

        # Pack all values as floats
        for row in value:
            for cell in row:
                result += pack("<f", float(cell))

        return result
```

### 2D Vector Schemas

#### Single 2D Vector Object

```python
from struct import pack
from typing import Any, Mapping

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
```

#### 2D Vector Array

```python
import numpy as np
from numbers import Real
from struct import pack
from typing import Any, Sequence

def _extract_vector_components(
    candidate: Any, expected_axes: Sequence[str]
) -> list[float] | None:
    """Convert a vector-like candidate into a list of floats when possible."""
    if isinstance(candidate, Mapping):
        try:
            return [float(candidate[axis]) for axis in expected_axes]
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)) and len(candidate) == len(expected_axes):
        vector_values: list[float] = []
        for element in candidate:
            if isinstance(element, Real):
                vector_values.append(float(element))
            else:
                return None
        return vector_values
    return None


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
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    normalized_vectors: list[list[float]] = []
    for vector_candidate in value:
        vector_components = _extract_vector_components(vector_candidate, expected_axes)
        if vector_components is None:
            return None
        normalized_vectors.append(vector_components)
    return normalized_vectors

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
```

### 3D Vector Schemas

#### Single 3D Vector Object

```python
from struct import pack
from typing import Any, Mapping

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
```

#### 3D Vector Array

```python
import numpy as np
from struct import pack
from typing import Any, Sequence

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
```

### 2D Pose Schemas

#### Single 2D Pose Object

```python
from struct import pack
from typing import Any, Mapping

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
            raise ValueError("Value must be a dict with x, y, and rotation keys containing numeric values")

        x = float(value["x"])
        y = float(value["y"])
        rotation = float(value["rotation"])

        return pack("<fff", x, y, rotation)
```

#### 2D Pose Array

```python
import numpy as np
from struct import pack
from typing import Any, Mapping, Sequence

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
                float(candidate["rotation"])
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
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    normalized_poses: list[list[float]] = []
    for pose_candidate in value:
        pose_components = _extract_pose2d_components(pose_candidate)
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
```

### 3D Pose Schemas

#### Single 3D Pose Object

```python
from struct import pack
from typing import Any, Mapping

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
            raise ValueError("Value must be a dict with x, y, z, roll, pitch, and yaw keys containing numeric values")

        x = float(value["x"])
        y = float(value["y"])
        z = float(value["z"])
        roll = float(value["roll"])
        pitch = float(value["pitch"])
        yaw = float(value["yaw"])

        return pack("<ffffff", x, y, z, roll, pitch, yaw)
```

#### 3D Pose Array

```python
import numpy as np
from struct import pack
from typing import Any, Mapping, Sequence

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
                float(candidate["yaw"])
            ]
        except (KeyError, TypeError, ValueError):
            return None
    return None

def _coerce_pose3d_sequence(value: Any) -> list[list[float]] | None:
    """Convert a sequence of 3D pose-like values into lists of floats."""
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[1] == 6:
            if np.issubdtype(value.dtype, np.floating):
                return value.tolist()
            elif np.issubdtype(value.dtype, np.number):
                return value.astype(float).tolist()
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    normalized_poses: list[list[float]] = []
    for pose_candidate in value:
        pose_components = _extract_pose3d_components(pose_candidate)
        if pose_components is None:
            return None
        normalized_poses.append(pose_components)
    return normalized_poses

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
```

## Advanced Patterns

### Type Coercion Helpers

```python
def _coerce_to_float(value: Any) -> float | None:
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def _normalize_vector(value: Any) -> list[float] | None:
    """Convert various vector formats to list of floats."""
    if isinstance(value, dict):
        # Handle {"x": 1, "y": 2} format
        x = _coerce_to_float(value.get("x"))
        y = _coerce_to_float(value.get("y"))
        if x is not None and y is not None:
            return [x, y]
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        # Handle [1, 2] format
        coords = [_coerce_to_float(v) for v in value]
        if all(c is not None for c in coords):
            return coords
    return None
```

### Validation Patterns

```python
def _validate_range(value: float, min_val: float, max_val: float) -> bool:
    """Check if value is within acceptable range."""
    return min_val <= value <= max_val

def _validate_string_length(string: str, max_length: int = 255) -> bool:
    """Check if string length is acceptable."""
    return len(string.encode("utf-8")) <= max_length
```

### Memory-Efficient Serialization

```python
def serialize_large_array(self, value: list[float]) -> bytes:
    """Serialize large arrays in chunks to reduce memory usage."""
    result = pack("<I", len(value))

    # Process in chunks of 1000 elements
    chunk_size = 1000
    for i in range(0, len(value), chunk_size):
        chunk = value[i:i + chunk_size]
        result += pack(f"<{len(chunk)}f", *chunk)

    return result
```

## Schema Naming Conventions

- Use lowercase with underscores: `my_custom_schema`
- Be descriptive but concise: `float_matrix` not `2d_float_array`
- Include dimensionality when relevant: `vector2`, `vector3`, `vector2_array`, `vector3_array`
- Use plural for arrays: `string_array`, `boolean_array`, `pose2d_array`, `pose3d_array`
- For single objects: use singular form: `vector2`, `vector3`, `pose2d`, `pose3d`
- Array schemas end with `_array`: `float_array`, `vector2_array`, `vector3_array`, `pose2d_array`, `pose3d_array`
- Single object schemas don't have `_array`: `vector2`, `vector3`, `pose2d`, `pose3d`

## Testing Your Schema

Create unit tests to verify your schema:

```python
def test_my_schema():
    schema = MyCustomSchema()

    # Test can_handle
    assert schema.can_handle([1, 2, 3])
    assert not schema.can_handle("not a list")

    # Test serialization round-trip (if applicable)
    original = [1, 2, 3]
    serialized = schema.serialize(original)
    # Implement deserialization to test completeness

    # Test edge cases
    assert schema.can_handle([])
    assert schema.can_handle([0, -1, 3.14])
```

## Performance Considerations

- **Minimize allocations**: Reuse buffers when possible
- **Use appropriate precision**: `f` vs `d` for float tradeoffs
- **Batch operations**: Group similar pack operations
- **Avoid string operations**: Use bytes throughout the pipeline

## Integration

Once your schema is created and tested:

1. Add array schemas to `array_schemas.py` and object schemas to `object_schemas.py`
2. Import and register it in `registry.py`
3. Update documentation and tests
4. Consider backward compatibility for existing data formats

**File Organization:**
- `array_schemas.py`: All schemas that handle sequences/arrays of data
- `object_schemas.py`: All schemas that handle single complex objects

## Common Pitfalls

- **Byte order**: Always use little-endian (`<`) for cross-platform compatibility
- **String encoding**: Use UTF-8 and include length prefixes
- **Null termination**: Avoid C-style null termination for strings
- **Alignment**: `struct` handles padding automatically
- **Type validation**: Be strict in `can_handle()` to avoid runtime errors
- **Error messages**: Provide clear error messages for debugging

## Schema Versioning

For future compatibility:

- Include version numbers in schema names: `vector3_array_v1`
- Document breaking changes
- Consider migration strategies for old formats
- Test against real NetworkTables consumers</contents>
</xai:function_call:>Write to file
