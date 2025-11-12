# Camera Adjust Operation

## Overview

The `CameraAdjust` is a secondary pipeline operation that provides hardware-accelerated camera parameter control for real-time image adjustment. It uses V4L2 (Video for Linux 2) controls to adjust camera settings like brightness, contrast, saturation, gain, and exposure directly at the hardware level, enabling optimal image capture for computer vision tasks.

## Operation Type

**Secondary Operation** - Hardware camera control utility

## Category

`util` - Utility operation

## Input/Output

- **Input**: `np.ndarray` (BGR image frame)
- **Output**: `np.ndarray` (same frame, hardware adjustments already applied)

### Processing Behavior

The operation passes frames through unchanged since adjustments are applied at the camera hardware level before image capture.

## Parameters

### Constructor Parameters

- `brightness` (float): Brightness offset, range [-1, 1], mapped to V4L2 range [-64, 64] (default: 0.0)
- `contrast` (float): Contrast multiplier, range [0, 1], mapped to V4L2 range [0, 64] (default: 0.5)
- `saturation` (float): Saturation multiplier, range [-1, 1], mapped to V4L2 range [0, 128] (default: 0.406)
- `gain` (float): Gain control, range [0, 1], mapped to V4L2 range [0, 100] (default: 0.0)
- `exposure` (float): Exposure time, range [0, 1], mapped to V4L2 range [1, 5000] (default: 0.5)
- `camera_manager` (Any): Injected camera manager reference (optional)
- `pipeline` (Any): Injected pipeline reference (optional)

## Configuration Example

### Pipeline Integration

```json
{
    "object_detection_pipeline": [
        {
            "action_name": "camera_adjust",
            "action_params": {
                "brightness": 0.2,
                "contrast": 0.7,
                "saturation": 0.3,
                "gain": 0.1,
                "exposure": 0.6
            }
        },
        {
            "action_name": "detect_apriltags",
            "action_params": {}
        }
    ]
}
```

### Python Usage Example

```python
from src.secondary_operations.camera_adjust import CameraAdjust

# Initialize camera adjustment with optimized settings
camera_adjust = CameraAdjust(
    brightness=0.2,    # Slightly brighter
    contrast=0.7,      # Higher contrast for better detection
    saturation=0.3,    # Moderate saturation
    gain=0.1,          # Low gain to reduce noise
    exposure=0.6       # Longer exposure for low light
)

# Apply settings to hardware (happens automatically on initialization)
# All subsequent frames will have these adjustments applied at camera level

frame = cv2.imread("input.jpg")
adjusted_frame = camera_adjust.run(frame)  # Frame already adjusted by hardware
```

## Performance Considerations

### Hardware Acceleration

- **Direct Hardware Control**: Adjustments applied at camera sensor level, no CPU overhead
- **Real-time Operation**: Settings change instantly without frame processing delays
- **Power Efficiency**: Hardware adjustments more efficient than software post-processing

### V4L2 Compatibility

- **Linux Systems**: Requires V4L2-compatible cameras (most USB/webcams)
- **Device Resolution**: Automatic device path resolution using camera manager
- **Error Handling**: Graceful fallback with error reporting for unsupported controls

### Parameter Ranges

- **Brightness**: Hardware-specific offset adjustment
- **Contrast**: Multiplier affecting dynamic range
- **Saturation**: Color intensity control
- **Gain**: Analog signal amplification
- **Exposure**: Integration time control (disables auto-exposure)

## Tuning Guide

### Brightness Adjustment

1. **Overexposed Scenes**: Decrease brightness (negative values)
2. **Underexposed Scenes**: Increase brightness (positive values)
3. **Default**: 0.0 for most lighting conditions

### Contrast Optimization

1. **Low Contrast**: Increase contrast (0.6-0.8) for better feature detection
2. **High Contrast**: Decrease contrast (0.3-0.5) for smoother gradation
3. **Detection Tasks**: Higher contrast often improves threshold-based detection

### Saturation Control

1. **Color Detection**: Moderate saturation (0.3-0.5) preserves color information
2. **Monochrome Tasks**: Low saturation (-0.5-0.0) for grayscale-like appearance
3. **Vibrant Scenes**: Higher saturation (0.5-1.0) enhances color differences

### Gain and Exposure

1. **Low Light**: Increase gain (0.2-0.5) for brighter images
2. **Motion Blur**: Shorter exposure (0.2-0.4) reduces blur but may darken image
3. **Static Scenes**: Longer exposure (0.6-0.8) for better low-light performance

### Auto vs Manual Control

- **Auto Mode**: Default camera auto-adjustment (exposure=0.5 leaves auto enabled)
- **Manual Mode**: Any exposure setting disables auto-exposure for consistent capture

## Use Cases

### AprilTag Detection Optimization

Hardware adjustments for reliable fiducial marker detection:

```json
{
    "brightness": 0.1,
    "contrast": 0.8,
    "saturation": 0.4,
    "gain": 0.0,
    "exposure": 0.7
}
```

### Color-Based Object Detection

Optimized settings for color threshold detection:

```json
{
    "brightness": 0.0,
    "contrast": 0.6,
    "saturation": 0.6,
    "gain": 0.2,
    "exposure": 0.5
}
```

### Low-Light Operation

Enhanced capture in challenging lighting conditions:

```json
{
    "brightness": 0.3,
    "contrast": 0.5,
    "saturation": 0.2,
    "gain": 0.4,
    "exposure": 0.8
}
```

## Limitations

1. **Hardware Support**: Requires V4L2-compatible cameras with adjustable controls
2. **Platform Specific**: Linux-only implementation using v4l2-ctl command
3. **Permission Requirements**: May need appropriate device access permissions
4. **Control Availability**: Not all camera controls may be supported by specific hardware
5. **No Software Fallback**: Purely hardware-based, no software image processing

## Visualization

The operation includes AprilTag detection visualization when back-propagated data is available:

### Features

- **Tag Outline Drawing**: Green polygons around detected AprilTag corners
- **Tag ID Labels**: White text displaying tag identification numbers
- **Center Positioning**: Labels positioned at tag geometric centers
- **Anti-aliased Rendering**: Smooth text and line rendering

### Usage

```python
camera_adjust = CameraAdjust(...)

# Back-propagate detections for visualization
camera_adjust.back_propagate_input(april_tag_detections)

# Visualize on frame
visualized_frame = camera_adjust.visualize(frame)
```

### Integration Pattern

```python
# Pipeline integration example
detections = april_tag_detector.run(frame)
camera_adjust.back_propagate_input(detections)
visualized_frame = camera_adjust.visualize(frame)
```

## Related Operations

- `DetectApriltagsDefinition`: Provides AprilTag detections for visualization
- `ColorThresholdDetectionDefinition`: Benefits from optimized camera settings
- `BackPropagateOperation`: Can feed detection data to camera adjust

## Files

- **Definition**: `src/secondary_operations/camera_adjust.py`
