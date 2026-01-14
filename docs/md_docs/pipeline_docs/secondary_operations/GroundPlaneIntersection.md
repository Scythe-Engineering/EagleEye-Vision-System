# GroundPlaneIntersection Operation

## Overview

The `GroundPlaneIntersection` operation is a secondary pipeline operation that calculates the 3D intersection points of detected objects with the ground plane. It uses camera pose and calibration parameters to estimate the real-world position of objects detected in 2D image space, enabling robot localization and navigation.

## Architecture

### Pinhole Camera Model

The operation implements a pinhole camera model for accurate projection:

1. **Bounding Box Extraction**: Extracts 2D bounding boxes from detection dictionaries
2. **Coordinate Normalization**: Converts pixel coordinates to normalized image coordinates
3. **Angle Calculation**: Computes horizontal and vertical viewing angles
4. **Ground Projection**: Projects rays onto the ground plane to obtain 3D positions

### Camera Parameter Integration

- **Camera Height**: Height of camera above ground plane (meters)
- **Camera Pitch**: Pitch angle of camera (radians, positive = looking down)
- **Field of View**: Horizontal and vertical FOV (degrees)

## Key Features

### Accurate 3D Position Estimation

- **Pinhole Model**: Uses accurate trigonometric calculations instead of linear scaling
- **Camera Pitch Compensation**: Accounts for camera tilt in calculations
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

- **camera_height**: Height of camera above ground plane in meters (default: 1.0)
- **camera_pitch**: Pitch angle of camera in radians (default: 0.0)
- **fov_horizontal**: Horizontal field of view in degrees (default: 60.0)
- **fov_vertical**: Vertical field of view in degrees (default: 45.0)

### Optional Parameters

- **pipeline**: Injected pipeline reference for accessing camera information

### Constructor

```python
def __init__(
    self,
    camera_height: float = 1.0,
    camera_pitch: float = 0.0,
    fov_horizontal: float = 60.0,
    fov_vertical: float = 45.0,
    pipeline: Any = None,
) -> None:
    """Initialize ground plane intersection operation.

    Args:
        camera_height: Height of camera above ground plane in meters
        camera_pitch: Pitch angle of camera in radians (positive = looking down)
        fov_horizontal: Horizontal field of view in degrees
        fov_vertical: Vertical field of view in degrees
        pipeline: Injected pipeline reference for accessing camera information
    """
```

## Data Flow

### Processing Flow

1. **Input Validation**: Check if detection is a dictionary with valid bounding box
2. **Coordinate Extraction**: Extract x_center and y_bottom from bounding box
3. **Angle Computation**: Calculate horizontal and vertical viewing angles
4. **Distance Calculation**: Compute distance using camera height and vertical angle
5. **3D Position**: Calculate x, y, z coordinates on ground plane
6. **Output Enrichment**: Add `position_3d` field to detection dictionary

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
Normalize coordinates to [-1, 1] range
       |
       v
Compute angles using pinhole model:
  - horizontal_angle_rad = atan(x_norm * tan(hfov/2))
  - vertical_angle_rad = atan(y_norm * tan(vfov/2)) + camera_pitch
       |
       v
Check minimum vertical angle (3°)
       |
       v
Calculate distance = camera_height / tan(vertical_angle)
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
        "camera_height": 0.5,
        "camera_pitch": 0.26,
        "fov_horizontal": 70.0,
        "fov_vertical": 50.0
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
        "camera_height": 0.5,
        "camera_pitch": 0.26,
        "fov_horizontal": 70.0,
        "fov_vertical": 50.0
      }
    }
  ]
}
```

### Runtime Configuration Update

```python
# Update camera parameters during runtime
ground_plane_op.update_config({
    "camera_height": 0.6,
    "camera_pitch": 0.30
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
horizontal_angle_rad = atan(x_norm_centered * tan(hfov_rad / 2.0))
where x_norm_centered = 2.0 * (x_center - 0.5) clipped to [-1, 1]
```

**Vertical Angle:**

```
vertical_angle_from_optical_rad = atan(y_norm_centered * tan(vfov_rad / 2.0))
total_vertical_angle_rad = vertical_angle_from_optical_rad + camera_pitch
where y_norm_centered = 2.0 * (y_bottom - 0.5) clipped to [-1, 1]
```

**Distance Calculation:**

```
distance = camera_height / tan(total_vertical_angle_rad)
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

- **Camera Calibration**: Requires accurate camera height, pitch, and FOV values
- **Undistorted Input**: Assumes input detections are from undistorted images
- **Ground Plane**: Assumes flat ground plane at z=0

### Performance Considerations

- **Per-Detection Processing**: Each detection processed independently
- **Trigonometric Operations**: Uses NumPy for efficient angle calculations
- **Caching Overhead**: Minimal overhead for thread-safe detection cache

### Calibration Guidelines

1. **Camera Height**: Measure from camera optical center to ground
2. **Camera Pitch**: Positive when camera tilts downward
3. **Field of View**: Use camera specification or calibration data
4. **Testing**: Validate with known distance measurements

## Error Handling

### Configuration Errors

- **Invalid Parameters**: Parameters are converted to float, may raise on non-numeric input
- **Missing Pipeline**: Optional pipeline reference, not required for operation

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
