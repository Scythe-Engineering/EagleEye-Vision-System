# DeviceInput Operation

## Overview

The `DeviceInput` operation is a source operation that fetches camera frames from the camera thread manager. It serves as the entry point for all vision pipelines, retrieving frames from a specified camera and passing them into the processing chain with capture timing metadata.

## Architecture

### Frame Source Pattern

The operation implements a camera frame fetching pattern:

1. **Camera Manager Integration**: Connects to the CameraThreadManager to access camera threads
2. **Packet Retrieval**: Calls `camera_manager.get_current_packet_by_bus_id()` to fetch the latest timed frame
3. **Frame Output**: Returns a `TimedValue` frame packet or None if unavailable

## Key Features

### Camera Integration

- **Camera Thread Manager Access**: Retrieves frames from active camera threads
- **Per-Camera Configuration**: Specifies which camera to read frames from via the deterministic `bus_id` parameter (USB port index)
- **Non-Blocking Retrieval**: Returns current packet if available, None otherwise

### Frame Data Handling

- **Timed Packet Output**: Returns `FramePacket` (`TimedValue[np.ndarray]`) for pipeline processing
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
2. **Frame Request**: `run()` method called to fetch current packet
3. **Camera Lookup**: Calls `camera_manager.get_current_packet_by_bus_id(bus_id)`
4. **Frame Return**: Returns timed frame packet or None if unavailable
5. **Pipeline Processing**: Returned packet enters next operation in pipeline

### Processing Steps

```
Camera Thread Manager
       |
       v
[DeviceInput.run()]
       |
       v
get_current_packet_by_bus_id(bus_id)
       |
       v
Return: FramePacket | None
       |
       v
[Next Operation in Pipeline]
```

## Usage Examples

### Basic Pipeline Configuration

```python
device_input = DeviceInput(
    camera_manager=camera_manager,
    bus_id="1",
)

packet = device_input.run(None)
if packet is not None:
    result = next_operation.run(packet)
```

### Camera Manager Integration

DeviceInput connects to CameraThreadManager which manages camera capture threads:

```python
camera_manager = CameraThreadManager(...)

packet = camera_manager.get_current_packet_by_bus_id("1")
if packet is not None:
    frame = packet.value
    capture_nt_us = packet.timing.capture_nt_us
```

## Directory Structure

```
src/secondary_operations/
└── device_input.py           # Main operation implementation
```

## Technical Details

### Input/Output Types

- **Input**: `Any` (unused - source operation doesn't require input)
- **Output**: `FramePacket | None` (timed video frame or None if unavailable)

### Method Signatures

```python
def run(self, input_data: Any) -> FramePacket | None:
    """Fetch the current frame from the configured camera.

    Args:
        input_data: Unused (data source operations don't use input).

    Returns:
        Current camera frame as a timed packet, or None if camera unavailable.
    """
    packet = self.camera_manager.get_current_packet_by_bus_id(self.bus_id)
    if packet is None:
        return None
    return TimedValue(self._apply_rotation(packet.value), packet.timing)
```

### Frame Data Format

- **Shape**: `(height, width, 3)` for color frames or `(height, width)` for grayscale
- **Data Type**: `np.uint8` typically (8-bit per channel)
- **Color Space**: BGR by default (OpenCV convention)
- **Timing**: `TimingMetadata.capture_nt_us` uses the NetworkTables microsecond timebase

## Integration Points

### Camera Manager Integration

- **CameraThreadManager**: Source for all frames in the system
- **Thread Safety**: Camera manager handles thread-safe frame access
- **Multiple Cameras**: Can create separate DeviceInput instances for different cameras

### Pipeline Integration

- **Source Operation**: Serves as the entry point for frame processing
- **Frame Provider**: Ensures timed frames available for all downstream operations
- **Pipeline Manager**: Created and managed by pipeline generation system

## Error Handling

### Camera Issues

- **Camera Not Found**: Returns None if bus_id doesn't match any registered camera
- **No Frames Available**: Returns None if camera thread hasn't captured a frame yet
- **Thread Not Running**: Returns None if camera thread is not active

### Robustness Features

- **Graceful None Handling**: Pipeline can handle None returns from DeviceInput
- **Non-Blocking**: Doesn't block pipeline waiting for frames

## Best Practices

### Pipeline Design

1. **Bus ID Registration**: Register bus IDs with CameraThreadManager before pipeline execution
2. **Single Source**: Assign only one camera per DeviceInput operation
3. **Error Handling**: Implement None checks in downstream operations

### Configuration

1. **Valid Bus IDs**: Ensure bus_id matches CameraThreadManager registration
2. **Camera Thread Setup**: Verify camera thread is running before pipeline execution
3. **Frame Timing**: Propagated capture timestamps are used for NetworkTables publishing
