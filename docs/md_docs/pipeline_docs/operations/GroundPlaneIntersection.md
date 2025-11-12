# Ground Plane Intersection Operation

## Overview

The `GroundPlaneIntersection` is a secondary pipeline operation that undistorts detection points using camera calibration parameters. It corrects for lens distortion in detected points using camera intrinsics and distortion coefficients.

## Operation Type

**Secondary Operation** - Point undistortion utility

## Category

`proc` - Processing operation

## Input/Output

- **Input**: `List[Dict[str, Any]]` - Detection dictionaries
- **Output**: `List[Dict[str, Any]]` - Detection dictionaries with undistorted information

## Parameters

### Constructor Parameters

- `intrinsics_path` (str): Path to camera calibration file or camera bus ID for auto-resolution
- `camera_height` (float): Camera height above ground plane in meters (default: 1.0)
- `camera_pitch` (float): Camera pitch angle in radians (default: 0.0, positive = looking down)
- `fov_horizontal` (float): Horizontal field of view in degrees (default: 60.0)
- `fov_vertical` (float): Vertical field of view in degrees (default: 45.0)
- `pipeline` (Any): Pipeline reference for camera information access (optional)

## Configuration Example

### Pipeline Integration

```json
{
    "object_detection_pipeline": [
        {
            "action_name": "color_threshold_detection",
            "action_params": {
                "color_ranges": [
                    {
                        "name": "target",
                        "class_id": 0,
                        "target_rgb": [1.0, 0.0, 0.0],
                        "threshold": 0.3
                    }
                ]
            }
        },
        {
            "action_name": "ground_plane_intersection",
            "action_params": {
                "intrinsics_path": "config/camera_intrinsics.json",
                "camera_height": 0.8,
                "camera_pitch": 0.1,
                "fov_horizontal": 60.0,
                "fov_vertical": 45.0
            }
        }
    ]
}
```

### Python Usage Example

```python
from src.secondary_operations.ground_plane_intersection import GroundPlaneIntersection

# Initialize with camera parameters
undistorter = GroundPlaneIntersection(
    intrinsics_path="config/camera_intrinsics.json",
    camera_height=0.8,     # 80cm above ground
    camera_pitch=0.1,      # Slight downward pitch
    fov_horizontal=60.0,   # 60 degree horizontal FOV
    fov_vertical=45.0      # 45 degree vertical FOV
)

# Example detections
detections = [
    {
        "bbox": [0.4, 0.3, 0.6, 0.7],
        "class_id": 0,
        "color_name": "red"
    }
]

# Process detections to undistort points
processed_detections = undistorter.run(detections)
```

## Performance Considerations

### Computational Load

- Minimal computational overhead for undistortion calculations
- Processes each detection independently
- Uses OpenCV undistortPoints for efficient distortion correction

### Robustness

- Thread-safe operation with proper locking
- Preserves all original detection metadata
- Graceful handling when distortion coefficients are not available

## Related Operations

- `ColorThresholdDetectionDefinition`: Provides 2D bounding box detections
- `ObjectDetectionDefinition`: Alternative source of 2D detections

## Files

- **Definition**: `src/secondary_operations/ground_plane_intersection.py`
