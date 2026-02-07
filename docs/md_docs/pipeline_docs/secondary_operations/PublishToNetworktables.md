# PublishToNetworktables Operation Overview

## Overview

The `PublishToNetworktables` operation is a secondary pipeline operation that publishes extracted data from the EagleEye object detection pipeline to NetworkTables. It uses Flatpack serialization to efficiently encode pipeline data as raw bytes, making it suitable for high-performance FRC robotics applications where minimizing serialization overhead and maximizing data throughput is critical.

## Architecture

### Flatpack Serialization System

The operation leverages a custom Flatpack serialization framework that wraps structured data in an efficient binary format:

- **Header**: `FPK1` magic bytes for format identification
- **Schema Metadata**: Embedded schema name for automatic deserialization
- **Binary Payload**: Compact binary representation of the data

### Supported Data Types

The Flatpack system currently supports three primary array types:

1. **Float Arrays**: Sequences of floating-point numbers
2. **Vector2 Arrays**: Arrays of 2D coordinate pairs (x, y)
3. **Vector3 Arrays**: Arrays of 3D coordinate triples (x, y, z)

### Data Path Extraction

The operation can extract specific values from complex pipeline data structures:

- **Full Data**: Publish entire pipeline output when no path is specified
- **Field Extraction**: Extract single fields from dictionaries
- **Sequence Field Extraction**: Extract the same field from all items in a list/array
- **Nested Access**: Navigate through nested structures using dot notation or array indices

## Key Features

### Efficient Serialization

- **Binary Format**: Flatpack uses compact binary encoding for minimal bandwidth usage
- **Schema-Aware**: Automatic type detection and schema selection
- **Extensible**: Easy to add new data types and schemas

### Flexible Data Selection

- **Path-Based Extraction**: Use `data_path` to specify which part of pipeline data to publish
- **List Field Extraction**: Special handling for extracting fields from all items in a list
- **Type-Safe Publishing**: Automatic type detection and appropriate NetworkTables method selection

### Real-Time Performance

- **Zero-Copy Forwarding**: Pipeline data passes through unchanged
- **Minimal Overhead**: Flatpack serialization optimized for speed
- **NetworkTables Integration**: Direct publishing to NT keys without intermediate processing

## Configuration

### Parameters and Dependencies

- **`target_key`**: NetworkTables entry key where data will be published
- **`data_path`**: Optional path to extract specific data from pipeline output
  (the operation only requires these parameters; any shared dependencies such as NetworkTables instances are injected automatically when the pipeline declares constructor parameters with those names) 
### Configuration Examples

#### Publishing Angles from Detection Results

```json
{
    "action_name": "publish_to_networktables",
    "action_params": {
        "target_key": "vision/target_angles",
        "data_path": ["angle_degrees"]
    }
}
```

#### Publishing Complete Detection Data

```json
{
    "action_name": "publish_to_networktables",
    "action_params": {
        "target_key": "vision/detections"
    }
}
```

#### Publishing 3D Positions

```json
{
    "action_name": "publish_to_networktables",
    "action_params": {
        "target_key": "vision/positions",
        "data_path": ["positions"]
    }
}
```

## Data Flow

### Input Processing

1. **Receive Pipeline Data**: Operation receives data from previous pipeline stage
2. **Path Resolution**: Apply `data_path` to extract desired values
3. **Special List Handling**: If extracting a field from a list of dicts, create array of extracted values
4. **Type Detection**: Determine appropriate Flatpack schema

### Serialization Process

1. **Schema Selection**: Choose appropriate Flatpack schema based on data type
2. **Binary Encoding**: Serialize data using schema-specific encoding
3. **Header Wrapping**: Add `FPK1` header and schema metadata
4. **NetworkTables Publishing**: Publish raw bytes to specified NT key

## Usage Examples

### Angle Tracking

For publishing horizontal angles to detected targets:

**Pipeline Data Structure:**
```python
[
    {"angle_degrees": 45.2, "area": 1000, "bbox": [0.1, 0.2, 0.3, 0.4]},
    {"angle_degrees": -12.8, "area": 800, "bbox": [0.5, 0.1, 0.7, 0.3]},
    {"angle_degrees": 15.7, "area": 1200, "bbox": [0.2, 0.6, 0.4, 0.8]}
]
```

**Configuration:**
```json
{
    "target_key": "vision/target_angles",
    "data_path": ["angle_degrees"]
}
```

**NetworkTables Result:** Raw bytes containing Flatpack-encoded float array `[45.2, -12.8, 15.7]`

### 3D Position Publishing

For publishing robot or object 3D positions:

**Pipeline Data Structure:**
```python
{
    "positions": [
        {"x": 1.2, "y": 3.4, "z": 0.1},
        {"x": 2.1, "y": -1.8, "z": 0.3},
        {"x": -0.5, "y": 2.2, "z": 0.0}
    ]
}
```

**Configuration:**
```json
{
    "target_key": "vision/object_positions",
    "data_path": ["positions"]
}
```

**NetworkTables Result:** Raw bytes containing Flatpack-encoded Vector3 array

## Directory Structure

```
secondary_operations/
├── publish_to_networktables.py              # Main operation implementation
├── config_data/
│   └── publish_to_networktables_config_def.json  # Configuration schema
└── flatpack_schema/                         # Flatpack serialization system
    ├── __init__.py
    ├── base.py                              # Base schema class
    ├── array_schemas.py                     # Array schema implementations
    └── registry.py                          # Schema registry and serialization
```

## Technical Details

### Flatpack Format Specification

**Binary Structure:**
```
[FPK1][Schema Name Length][Schema Name][Payload Length][Payload Data]
```

- **FPK1**: 4-byte magic identifier
- **Schema Name Length**: 4-byte unsigned integer
- **Schema Name**: UTF-8 encoded schema name
- **Payload Length**: 4-byte unsigned integer
- **Payload Data**: Schema-specific binary data

### Schema Implementations

#### FloatArraySchema
- **Data Type**: `list[float]`
- **Encoding**: IEEE 754 double-precision floats in native byte order
- **NetworkTables Type**: Raw bytes

#### Vector2ArraySchema
- **Data Type**: `list[tuple[float, float]]`
- **Encoding**: Interleaved X,Y coordinates as doubles
- **NetworkTables Type**: Raw bytes

#### Vector3ArraySchema
- **Data Type**: `list[tuple[float, float, float]]`
- **Encoding**: Interleaved X,Y,Z coordinates as doubles
- **NetworkTables Type**: Raw bytes

## Integration Points

### Pipeline Integration

- **Input**: Any pipeline data structure
- **Output**: Unmodified pipeline data (pass-through operation)
- **Dependencies**: NetworkTables instance injected by pipeline framework

### NetworkTables Integration

- **Publishing Method**: `putRaw()` for binary Flatpack data
- **Key Management**: User-specified target keys
- **Type Safety**: Schema metadata ensures proper deserialization

### Consumer Requirements

NetworkTables consumers must:

1. **Header Detection**: Check for `FPK1` magic bytes
2. **Schema Reading**: Extract and parse schema name
3. **Payload Decoding**: Use schema-appropriate deserialization
4. **Type Safety**: Handle schema version compatibility

## Development Notes

### Adding New Schemas

To add support for new data types:

1. Create new schema class inheriting from `FlatpackSchema`
2. Implement `serialize()` method with binary encoding logic
3. Register schema in `FlatpackRegistry`
4. Update operation documentation

### Performance Considerations

- **Memory Usage**: Flatpack minimizes allocations during serialization
- **Network Efficiency**: Binary format reduces bandwidth compared to JSON/NT native types
- **CPU Overhead**: Schema-based serialization optimized for common FRC data patterns

## Error Handling

### Serialization Failures

- **Unsupported Types**: Raises `ValueError` when data doesn't match any schema
- **Invalid Paths**: Returns `None` for unresolvable data paths
- **Schema Conflicts**: Registry raises errors for duplicate schema names

### NetworkTables Errors

- **Connection Issues**: NetworkTables handles connection failures gracefully
- **Type Conflicts**: Raw bytes avoid NT type validation conflicts
- **Key Conflicts**: Overwrites existing keys without validation

## Future Enhancements

### Planned Schema Additions

- **String Arrays**: For publishing lists of text data
- **Boolean Arrays**: For publishing boolean state arrays
- **Matrix Arrays**: For publishing transformation matrices
- **Compressed Arrays**: Run-length encoding for repetitive data

### Protocol Extensions

- **Version Negotiation**: Schema version compatibility checking
- **Compression Options**: Optional payload compression for large datasets
- **Streaming Support**: Incremental publishing for very large arrays</contents>
</xai:function_call:>Write to file
