# CameraAdjust Operation Overview

## Overview

The `CameraAdjust` operation is a secondary pipeline operation that provides hardware-accelerated camera parameter control through the Video4Linux2 (v4l2) subsystem. This operation enables real-time adjustment of camera settings like brightness, contrast, saturation, gain, and exposure directly at the hardware level for optimal vision processing performance.

## Architecture

### Hardware Control System

The operation implements direct camera hardware manipulation through:

1. **V4L2 Integration**: Uses `v4l2-ctl` command-line tool for hardware control
2. **Parameter Mapping**: Converts normalized parameters to device-specific ranges
3. **Device Discovery**: Resolves camera device paths from the operation's `camera_bus_id` and the camera manager
4. **Change Detection**: Applies settings only when parameters actually change

### Visualization Integration

The operation includes AprilTag detection visualization capabilities, receiving back-propagated detection data to overlay visual feedback on camera adjustment settings.

## Key Features

### Hardware-Accelerated Controls

- **Brightness Control**: Hardware-level brightness adjustment (-1.0 to 1.0 range)
- **Contrast Adjustment**: Real-time contrast modification (0.0 to 1.0 range)
- **Saturation Control**: Color saturation tuning (-1.0 to 1.0 range)
- **Gain Management**: Electronic gain adjustment (0.0 to 1.0 range)
- **Exposure Control**: Automatic exposure override with manual timing (0.0 to 1.0 range)

### Intelligent Application

- **Change Detection**: Only applies settings when values actually change
- **Device Auto-Discovery**: Automatically finds correct camera device paths
- **Error Handling**: Graceful degradation when hardware control fails
- **Normalized Interface**: User-friendly 0-1 parameter ranges

### Visualization Features

- **AprilTag Overlay**: Visual feedback showing detected AprilTags
- **Back-Propagation Support**: Receives detection data from downstream operations
- **Real-Time Display**: Live visualization of detection results on adjusted frames

## Configuration

### Primary Parameters

- **camera_bus_id**: Deterministic camera ID (same as `device_input` / calibration) used to resolve the v4l2 device
- **brightness**: Brightness offset (-1.0 to 1.0, default: 0.0)
- **contrast**: Contrast multiplier (0.0 to 1.0, default: 0.5)
- **saturation**: Saturation multiplier (-1.0 to 1.0, default: 0.406)
- **gain**: Gain control (0.0 to 1.0, default: 0.0)
- **exposure**: Exposure time (0.0 to 1.0, default: 0.5)

### System Integration

- **camera_manager**: Camera manager reference for device resolution

### Configuration Example

```python
camera_adjust = CameraAdjust(
    camera_bus_id="0",
    brightness=0.2,
    contrast=0.7,
    saturation=0.3,
    gain=0.1,
    exposure=0.8,
    camera_manager=camera_mgr,
)
```

## Data Flow

### Processing Flow

1. **Frame Reception**: Accept input video frame
2. **Hardware Application**: Camera settings already applied at hardware level
3. **Visualization Enhancement**: Add AprilTag detection overlays if available
4. **Frame Return**: Return processed frame with visual annotations

### Processing Steps

```
Input: Video Frame
       ↓
Hardware adjustments already active
       ↓
If AprilTag detections available:
  Overlay detection visualizations
       ↓
Return enhanced frame
```

## Usage Examples

### Basic Camera Adjustment

```python
# Configure camera for low-light conditions
low_light_adjust = CameraAdjust(
    camera_bus_id="0",
    brightness=0.3,
    contrast=0.8,
    gain=0.6,
    exposure=0.9,
    camera_manager=camera_manager,
)

# Process frame with adjusted camera settings
adjusted_frame = low_light_adjust.run(input_frame)
```

### Dynamic Adjustment Pipeline

```json
{
  "operations": [
    {
      "type": "secondary",
      "name": "camera_adjust",
      "config": {
        "brightness": 0.1,
        "contrast": 0.6,
        "exposure": 0.7
      }
    },
    {
      "type": "primary",
      "name": "apriltag_detection"
    },
    {
      "type": "secondary",
      "name": "back_propagate",
      "config": {
        "action_name": "camera_adjust"
      }
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
└── camera_adjust.py    # Main operation implementation
```

## Technical Details

### V4L2 Parameter Mapping

**Brightness Mapping:**
```
v4l2_value = int(normalized_brightness * 64)  # Range: -64 to 64
```

**Contrast Mapping:**
```
v4l2_value = int(normalized_contrast * 64)     # Range: 0 to 64
```

**Saturation Mapping:**
```
v4l2_value = int((normalized_saturation + 1) * 64)  # Range: 0 to 128
```

**Gain Mapping:**
```
v4l2_value = int(normalized_gain * 100)        # Range: 0 to 100
```

**Exposure Mapping:**
```
v4l2_value = int(1 + (normalized_exposure * 4999))  # Range: 1 to 5000
```

### Device Path Resolution

**Automatic Discovery** (via configured `camera_bus_id` and `camera_manager`):

```python
bus_id = operation.camera_bus_id
camera_name = camera_manager.get_camera_name_by_bus_id(bus_id)
worker = camera_manager.cameras.get(camera_name)
device_path = f"/dev/video{int(worker.camera.camera_index)}"
```

### Visualization Implementation

- **Corner Detection**: Draws AprilTag corner polygons
- **ID Annotation**: Displays tag IDs at detection centers
- **Color Coding**: Uses green outlines for detected tags

## Integration Points

### Pipeline Integration

- **Early Placement**: Should be placed early in pipeline for hardware effect
- **Back-Propagation**: Receives detection data for visualization feedback
- **Hardware Synchronization**: Coordinates with camera hardware timing

### Camera System Integration

- **Device Management**: Works with camera manager for device enumeration
- **Bus ID Resolution**: Uses camera bus IDs for device identification
- **Hardware Compatibility**: Supports Video4Linux2-compatible cameras

## Development Notes

### Hardware Requirements

- **V4L2 Support**: Cameras must support Video4Linux2 controls
- **v4l2-ctl Tool**: Command-line tool must be installed on system
- **Device Permissions**: Appropriate permissions for camera device access

### Performance Considerations

- **Hardware Acceleration**: Settings applied at hardware level (no CPU overhead)
- **Change Detection**: Avoids redundant hardware control calls
- **Command Execution**: Subprocess calls for v4l2-ctl with timeout protection

## Error Handling

### Hardware Control Failures

- **Command Execution**: Handles v4l2-ctl command failures gracefully
- **Device Access**: Manages cases where camera devices are unavailable
- **Parameter Validation**: Ensures parameter values are within valid ranges

### Robustness Features

- **Fallback Behavior**: Continues operation when hardware control fails
- **Error Logging**: Detailed error messages for troubleshooting
- **Timeout Protection**: Prevents hanging on command execution

## Future Enhancements

### Planned Features

- **Auto-Adjustment**: Automatic parameter optimization based on image analysis
- **Profile Management**: Save/load camera parameter profiles
- **Temperature Compensation**: Automatic adjustment for temperature effects
- **Scene Detection**: Adaptive settings based on lighting conditions
- **Multi-Camera Support**: Coordinated adjustment across multiple cameras
- **Parameter Interpolation**: Smooth transitions between parameter sets
