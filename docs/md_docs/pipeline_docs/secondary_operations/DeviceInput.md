# DeviceInput Operation

## Overview

The `DeviceInput` operation is a source operation that fetches camera frames from the camera thread manager. It serves as the entry point for all vision pipelines, retrieving frames from a specified camera and passing them into the processing chain for analysis.

## Architecture

### Frame Source Pattern

The operation implements a camera frame fetching pattern:

1. **Camera Manager Integration**: Connects to the CameraThreadManager to access camera threads
2. **Frame Retrieval**: Calls `camera_manager.get_current_frame()` to fetch the latest frame
3. **Frame Output**: Returns the frame as a numpy array or None if unavailable

## Key Features

### Camera Integration

- **Camera Thread Manager Access**: Retrieves frames from active camera threads
- **Per-Camera Configuration**: Specifies which camera to read frames from via the deterministic `bus_id` parameter (USB port index)
- **Non-Blocking Retrieval**: Returns current frame if available, None otherwise

### Frame Data Handling

- **Numpy Array Output**: Returns frames as `np.ndarray` for pipeline processing
- **Graceful None Handling**: Returns None when camera is unavailable or no frame ready
- **Type Safety**: Proper type hints for input/output data

## Configuration

### Required Parameters

- **bus_id** (`str`): USB bus ID of the camera to fetch frames from (matches `camera_manager` registration)

### Constructor

```python
def __init__(self, camera_manager: CameraThreadManager, bus_id: str) -> None:
    """Initialize the device input operation.

    Args:
        camera_manager: Camera thread manager to fetch frames from.
        bus_id: Deterministic bus ID of the camera to read frames from.
    """
```

## Data Flow

### Processing Flow

1. **Initialization**: DeviceInput is created with reference to camera manager and camera bus ID
2. **Frame Request**: `run()` method called to fetch current frame
3. **Camera Lookup**: Calls `camera_manager.get_current_frame_by_bus_id(bus_id)`
4. **Frame Return**: Returns numpy array frame or None if unavailable
5. **Pipeline Processing**: Returned frame enters next operation in pipeline

### Processing Steps

```
Camera Thread Manager
       |
       v
[DeviceInput.run()]
       |
       v
get_current_frame(camera_name)
       |
       v
Return: np.ndarray | None
       |
       v
[Next Operation in Pipeline]
```

## Usage Examples

### Basic Pipeline Configuration

```python
# DeviceInput is typically created by the pipeline manager
device_input = DeviceInput(
    camera_manager=camera_manager,
    camera_name="Camera_0"
)

# Used in pipeline execution
frame = device_input.run(None)  # input_data is unused for source operations
if frame is not None:
    # Frame available, pass to next operation
    result = next_operation.run(frame)
```

### Camera Manager Integration

DeviceInput connects to CameraThreadManager which manages camera capture threads:

```python
# Camera manager maintains active camera threads
camera_manager = CameraThreadManager(...)

# DeviceInput retrieves frames from active threads
frame_result = camera_manager.get_current_frame("Camera_0")
if frame_result is not None:
    frame, timestamp = frame_result
    # frame is np.ndarray of current camera image
```

## Directory Structure

```
src/secondary_operations/
└── device_input.py           # Main operation implementation
```

## Technical Details

### Input/Output Types

- **Input**: `Any` (unused - source operation doesn't require input)
- **Output**: `np.ndarray | None` (video frame or None if unavailable)

### Method Signatures

```python
def run(self, input_data: Any) -> np.ndarray | None:
    """Fetch the current frame from the configured camera.

    Args:
        input_data: Unused (data source operations don't use input).

    Returns:
        Current camera frame as numpy array, or None if camera unavailable.
    """
    frame_result = self.camera_manager.get_current_frame(self.camera_name)
    if frame_result is not None:
        frame, _ = frame_result
        return frame
    return None
```

### Frame Data Format

- **Shape**: `(height, width, 3)` for color frames or `(height, width)` for grayscale
- **Data Type**: `np.uint8` typically (8-bit per channel)
- **Color Space**: BGR by default (OpenCV convention)
- **Timestamp**: Included in camera manager result but not returned by DeviceInput

## Integration Points

### Camera Manager Integration

- **CameraThreadManager**: Source for all frames in the system
- **Thread Safety**: Camera manager handles thread-safe frame access
- **Multiple Cameras**: Can create separate DeviceInput instances for different cameras

### Pipeline Integration

- **Source Operation**: Serves as the entry point for frame processing
- **Frame Provider**: Ensures frames available for all downstream operations
- **Pipeline Manager**: Created and managed by pipeline generation system

## Development Notes

### Operation Requirements

- **Camera Availability**: Camera must be registered with CameraThreadManager before use
- **Camera Thread Active**: Camera thread must be running to return frames
- **Name Matching**: `camera_name` parameter must match registered camera name exactly

### Pipeline Design

- **Single Source**: Each pipeline has one DeviceInput for its assigned camera
- **Required Operation**: Every pipeline must have a DeviceInput operation
- **First in Chain**: Always executes before other operations in pipeline

## Error Handling

### Camera Issues

- **Camera Not Found**: Returns None if camera_name doesn't match any active camera
- **No Frames Available**: Returns None if camera thread hasn't captured a frame yet
- **Thread Not Running**: Returns None if camera thread is not active

### Robustness Features

- **Graceful None Handling**: Pipeline can handle None returns from DeviceInput
- **Non-Blocking**: Doesn't block pipeline waiting for frames
- **Safe Tuple Unpacking**: Properly handles frame_result tuple structure

## Best Practices

### Pipeline Design

1. **Camera Naming**: Use clear, consistent camera names across configuration
2. **Single Source**: Assign only one camera per DeviceInput operation
3. **Error Handling**: Implement None checks in downstream operations

### Configuration

1. **Valid Camera Names**: Ensure camera_name matches CameraThreadManager registry
2. **Camera Thread Setup**: Verify camera thread is running before pipeline execution
3. **Frame Timing**: Consider camera frame rate when designing pipeline timing

## Future Enhancements

### Planned Features

- **Frame Buffering**: Optional frame buffer for frame averaging/history
- **Frame Validation**: Format and dimension validation before return
- **Metadata Inclusion**: Timestamp and camera metadata in frame data
- **Multi-Camera Frames**: Support for synchronized multi-camera frame sets
- **Frame Skipping**: Optional frame rate limiting or skipping

---

_Last Updated: January 2025_
