# GroundPlaneIntersection Operation

## Overview

The `GroundPlaneIntersection` operation is a secondary pipeline operation that calculates the 3D intersection points of detected objects with the ground plane. It uses camera pose and calibration parameters to estimate the real-world position of objects detected in 2D image space, enabling robot localization and navigation.

## Architecture

### Pinhole Camera Model

The operation implements a pinhole camera model for accurate projection:

1. **Bounding Box Extraction**: Extracts 2D bounding boxes from detection dictionaries
2. **Intrinsics Lookup**: Loads the selected camera's intrinsics from the camera config registry
3. **Ray Calculation**: Converts normalized image coordinates into camera rays using `camera_matrix`
4. **Ground Projection**: Projects rays onto the ground plane to obtain 3D positions

### Camera Parameter Integration

- **Camera Height**: Uses the selected camera extrinsics `z_offset` value (meters)
- **Camera Pitch**: Uses the selected camera extrinsics `pitch` value (degrees, converted to radians)
- **Camera Intrinsics**: Selected camera bus ID resolves `camera_matrix` and image size

## Key Features

### Accurate 3D Position Estimation

- **Pinhole Model**: Uses accurate trigonometric calculations instead of linear scaling
- **Camera Extrinsics Integration**: Reads pitch and height from the camera config editor
- **Ground Plane Projection**: Projects intersection points onto z=0 plane

### Robust Processing

- **Invalid Detection Filtering**: Skips detections without valid bounding boxes
- **Minimum Angle Threshold**: Ignores detections too close to horizon (3° minimum)
- **Finite Value Checks**: Validates all computed values before output

### Thread-Safe Caching

- **Last Detections Storage**: Maintains thread-safe cache of recent detections
- **Lock-Based Access**: Uses threading locks for concurrent access
- **Runtime Configuration**: Supports dynamic parameter updates via `update_config()`

## Configuration

### Required Parameters

- **camera_bus_id**: Camera USB bus ID used to resolve intrinsics

### Optional Parameters

- **camera_config_registry**: Injected registry for resolving camera intrinsics and extrinsics

### Constructor

```python
def __init__(
    self,
    camera_bus_id: str | None = None,
    camera_height: float = 1.0,
    camera_pitch: float = 0.0,
    camera_config_registry: CameraConfigRegistry | None = None,
) -> None:
    """Initialize ground plane intersection operation.

    Args:
        camera_bus_id: Camera bus ID used to resolve intrinsics.
        camera_height: Legacy fallback height used when extrinsics are unavailable.
        camera_pitch: Legacy fallback pitch used when extrinsics are unavailable.
        camera_config_registry: Injected shared camera config registry.
    """
```

## Data Flow

### Processing Flow

1. **Input Validation**: Check if detection is a dictionary with valid bounding box
2. **Intrinsics Lookup**: Load camera matrix and image size for the selected camera
3. **Extrinsics Lookup**: Load `z_offset` and `pitch` for the selected camera
4. **Coordinate Extraction**: Extract x_center and y_bottom from bounding box
5. **Ray Projection**: Convert normalized image coordinates into pinhole camera rays
6. **Distance Calculation**: Compute distance using camera height and vertical angle
7. **3D Position**: Calculate x, y, z coordinates on ground plane
8. **Output Enrichment**: Add `position_3d` field to detection dictionary

### Processing Steps

```
Input: List[Dict[str, Any]] with detection information
       |
       v
For each detection:
  - Extract bounding box (x1, y1, x2, y2)
  - Calculate x_center = (x1 + x2) / 2
  - Calculate y_bottom = max(y1, y2)
       |
       v
Convert normalized coordinates to pixel coordinates
       |
       v
Compute angles using selected camera intrinsics:
  - x_ray = (x_pixel - cx) / fx
  - y_ray = (y_pixel - cy) / fy
  - horizontal_angle_rad = atan(x_ray)
  - vertical_angle_rad = atan(y_ray) + radians(extrinsics.pitch)
       |
       v
Check minimum vertical angle (3°)
       |
       v
Calculate distance = extrinsics.z_offset / tan(vertical_angle)
       |
       v
Compute 3D position:
  - x_position = distance * cos(horizontal_angle)
  - y_position = distance * sin(horizontal_angle)
  - z_position = 0.0
       |
       v
Output: List[Dict[str, Any]] with position_3d field
```

## Usage Examples

### Basic Configuration

```json
{
    "action_name": "ground_plane_intersection",
    "action_params": {
        "camera_bus_id": "0"
    }
}
```

### Pipeline Integration

```python
# In pipeline configuration
{
  "operations": [
    {
      "type": "main",
      "name": "object_detection"
    },
    {
      "type": "secondary",
      "name": "ground_plane_intersection",
      "config": {
        "camera_bus_id": "0"
      }
    }
  ]
}
```

### Runtime Configuration Update

```python
# Update camera parameters during runtime
ground_plane_op.update_config({
    "camera_bus_id": "1"
})
```

## Directory Structure

```
src/secondary_operations/
 ground_plane_intersection.py    # Main operation implementation
```

## Technical Details

### Input/Output Types

- **Input**: `List[Dict[str, Any]]` - List of detection dictionaries
- **Output**: `List[Dict[str, Any]]` - List of detections with `position_3d` field

### Detection Dictionary Format

**Input Detection:**

```python
{
    "bbox": [x1, y1, x2, y2],  # Normalized coordinates [0, 1]
    # ... other detection fields
}
```

**Output Detection:**

```python
{
    "bbox": [x1, y1, x2, y2],
    "position_3d": [y_position, z_position, x_position],  # [meters]
    # ... other detection fields
}
```

### Coordinate System

- **Input**: Normalized image coordinates [0, 1]
- **Output**: 3D world coordinates in meters
- **Ground Plane**: z = 0.0
- **Position Format**: `[y_position, z_position, x_position]`

### Mathematical Model

**Horizontal Angle:**

```
x_pixel = clip(x_center, 0.0, 1.0) * image_width
x_ray = (x_pixel - cx) / fx
horizontal_angle_rad = atan(x_ray)
```

**Vertical Angle:**

```
y_pixel = clip(y_bottom, 0.0, 1.0) * image_height
y_ray = (y_pixel - cy) / fy
vertical_angle_from_optical_rad = atan(y_ray)
total_vertical_angle_rad = vertical_angle_from_optical_rad + radians(extrinsics.pitch)
```

**Distance Calculation:**

```
distance = extrinsics.z_offset / tan(total_vertical_angle_rad)
```

**3D Position:**

```
x_position = distance * cos(horizontal_angle_rad)
y_position = distance * sin(horizontal_angle_rad)
z_position = 0.0
```

## Integration Points

### Pipeline Integration

- **After Detection**: Typically placed after object detection operations
- **Before Localization**: Often used before robot localization or navigation operations
- **Multi-Camera**: Can be used with multiple cameras for position estimation

### Operation Interface

- **Optional Methods**: Implements `update_config()` for runtime parameter changes
- **No Visualization**: Does not implement `visualize()` method
- **Thread-Safe**: Uses locks for detection cache access

## Development Notes

### Operation Requirements

- **Camera Calibration**: Requires accurate camera intrinsics and extrinsics editor values
- **Undistorted Input**: Assumes input detections are from undistorted images
- **Ground Plane**: Assumes flat ground plane at z=0

### Performance Considerations

- **Per-Detection Processing**: Each detection processed independently
- **Trigonometric Operations**: Uses NumPy for efficient angle calculations
- **Caching Overhead**: Minimal overhead for thread-safe detection cache

### Calibration Guidelines

1. **Camera Height**: Set `Z Offset` to the camera optical center height above ground
2. **Camera Pitch**: Set `Pitch` in the camera extrinsics editor
3. **Intrinsics**: Upload the camera calibration JSON for the selected bus ID
4. **Testing**: Validate with known distance measurements

## Error Handling

### Configuration Errors

- **Invalid Parameters**: Parameters are converted to float, may raise on non-numeric input
- **Missing camera_bus_id**: Operation cannot resolve a v4l2 device without a configured camera ID

### Runtime Errors

- **Missing Bounding Box**: Skips detections without valid `bbox` field
- **Invalid Coordinates**: Skips detections with invalid coordinate values
- **Minimum Angle**: Ignores detections below 3° vertical angle threshold
- **Non-Finite Values**: Skips detections producing non-finite distance or positions

## Best Practices

### Pipeline Design

1. **After Detection**: Place after object detection or AprilTag detection operations
2. **Before Localization**: Use before robot localization or navigation operations
3. **Camera-Specific**: May need different parameters for each camera

### Configuration

1. **Accurate Calibration**: Use precise camera measurements
2. **Test Validation**: Validate with known distance measurements
3. **Parameter Tuning**: Adjust minimum angle threshold based on use case

### Performance

1. **Filter Early**: Filter detections before ground plane intersection to reduce processing
2. **Batch Processing**: Process multiple detections in single operation call
3. **Cache Results**: Use the built-in detection cache for downstream operations

## Future Enhancements

### Planned Features

- **Multi-Plane Support**: Support for non-horizontal planes
- **Camera Roll/Yaw**: Additional camera orientation parameters
- **Confidence Scoring**: Add confidence scores based on angle and distance
- **Visualization**: Optional visualization of projected rays and intersection points
- **Calibration Tools**: Built-in calibration utilities for parameter estimation
- **Performance Metrics**: Timing and accuracy metrics for monitoring

---

_Last Updated: January 2025_
