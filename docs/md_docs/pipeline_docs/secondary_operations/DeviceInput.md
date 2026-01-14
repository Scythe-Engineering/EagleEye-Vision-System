# DeviceInput Operation

## Overview

The `DeviceInput` operation is a special secondary pipeline operation that serves as the designated entry point for all vision pipelines. It marks where camera frames enter the processing chain and is automatically detected by the flow manager as the starting operation.

## Architecture

### Pipeline Entry Point

The operation implements a simple pass-through pattern:

1. **Flow Manager Detection**: The flow manager searches for an operation named `"device_input"` to identify the pipeline start
2. **Frame Injection**: Camera frames are injected directly into the next operation in the chain
3. **Pass-Through**: If called directly, returns the input frame unchanged

## Key Features

### Automatic Discovery

- **Name-Based Detection**: Flow manager locates the operation by name (`"device_input"`)
- **Pipeline Start Point**: Always the first operation in execution order
- **Error Handling**: Raises `ValueError` if no device_input operation is found

### Frame Injection Pattern

- **Direct Injection**: Frames bypass the `run()` method and are injected into subsequent operations
- **Warning on Direct Call**: Logs a warning if `run()` is called (should not happen in normal operation)
- **No Processing**: Pure pass-through operation with no transformation logic

## Configuration

### Required Parameters

None - the operation requires no configuration parameters.

### Constructor

```python
def __init__(self) -> None:
    """Initialize the device input operation."""
```

## Data Flow

### Processing Flow

1. **Pipeline Initialization**: Flow manager finds the device_input operation
2. **Frame Injection**: Camera frames are injected into the next connected operation
3. **Processing Continues**: Frame passes through the pipeline normally

### Processing Steps

```
Camera Frame
       |
       v
[DeviceInput] (entry point marker)
       |
       v
[Next Operation] (frame injected directly)
       |
       v
...
```

## Usage Examples

### Basic Pipeline Configuration

```json
{
    "operations": [
        {
            "type": "secondary",
            "name": "device_input",
            "position": {
                "x": 100,
                "y": 100
            },
            "connections": [
                {
                    "from_uuid": "device_input_uuid",
                    "from_port": "frame",
                    "to_uuid": "next_operation_uuid",
                    "to_port": "frame",
                    "data_type": "frame",
                    "is_default": false
                }
            ]
        }
    ]
}
```

### Flow Manager Integration

The flow manager automatically finds the device_input operation:

```python
def _find_start_operation(self) -> Operation:
    """Finds the starting operation in the flow, always is the device_input operation name."""
    for uuid, operation_data in self.operations.items():
        if operation_data.name == "device_input":
            return self.operations[uuid]
    raise ValueError("No starting operation (device_input) found in operations.")
```

## Directory Structure

```
src/secondary_operations/
 device_input.py           # Main operation implementation
```

## Technical Details

### Input/Output Types

- **Input**: `Any` (typically `np.ndarray` for camera frames)
- **Output**: `Any` (returns input unchanged)

### Method Signatures

```python
def run(self, frame: Any) -> Any:
    """Return the input frame unchanged. Should not be used, but if it is do not error.

    Args:
        frame: Input camera frame.

    Returns:
        The input frame.
    """
    print(
        "DeviceInput.run() should not be called during normal operation, "
        "frame should be injected into next operations instead."
    )
    return frame
```

## Integration Points

### Pipeline Integration

- **Flow Manager**: Automatically detects device_input as pipeline start
- **Camera Manager**: Injects frames into the pipeline at the device_input operation
- **Visual Editor**: Displays as the first node in the pipeline flowchart

### Operation Interface

- **No Special Methods**: Does not implement `visualize()` or `update_config()`
- **Simple Pass-Through**: Returns input unchanged when run directly

## Development Notes

### Operation Requirements

- **Required Name**: Must be named `"device_input"` for flow manager detection
- **No Configuration**: Takes no parameters in constructor
- **Minimal Implementation**: Simple pass-through logic

### Pipeline Design

- **Always Present**: Every pipeline must have exactly one device_input operation
- **Position**: Typically placed at the leftmost position in visual editor
- **Connection**: Must connect to at least one downstream operation

## Error Handling

### Configuration Errors

- **Missing Operation**: Raises `ValueError` if no device_input operation found
- **Multiple Operations**: Flow manager uses the first match (though pipelines should have only one)

### Runtime Warnings

- **Direct Call**: Logs warning if `run()` method is called directly (indicates incorrect pipeline setup)

## Best Practices

### Pipeline Design

1. **Single Entry Point**: Each pipeline should have exactly one device_input operation
2. **Visual Placement**: Position at the left edge of the canvas for clear flow visualization
3. **Connection Pattern**: Connect output to the first processing operation

### Configuration

1. **No Parameters**: Do not add configuration parameters to device_input
2. **Static Position**: Position should remain stable across pipeline edits
3. **Clear Label**: Use "Device Input" or similar clear label in visual editor

## Future Enhancements

### Planned Features

- **Multi-Input Support**: Potential support for multiple camera inputs per pipeline
- **Frame Metadata**: Addition of frame metadata injection (timestamp, camera ID, etc.)
- **Validation**: Frame format validation before injection
- **Debug Mode**: Optional frame logging for debugging

---

_Last Updated: January 2025_
