# YTD Camera Localization Operation

## Overview

The `YtdCameraLocalizationDefinition` is a main pipeline operation that determines camera position and orientation in 3D space by detecting AprilTags using a specialized Yaw-Tag-Distance (YTD) localization algorithm. This method provides robust pose estimation particularly effective for ground-based robot navigation.

## Operation Type

**Main Operation** - Uses implementation in `src/main_operations/modules/apriltags/ytd_camera_localization/ytd_localization.py`

## Category

`loc` - Localization operation

## Input/Output

- **Input**: `List[ApriltagDetection]` - AprilTag detection objects with pose information
- **Output**: `np.ndarray` or `None` - 4x4 transformation matrix representing camera pose in global coordinates

### Pose Matrix Format

The output transformation matrix is a 4x4 numpy array where:
- **Rotation**: Top-left 3x3 submatrix representing camera orientation
- **Translation**: Top-right 3x1 vector representing camera position in meters
- **Homogeneous**: Bottom row is [0, 0, 0, 1] for matrix multiplication compatibility

## Processing Pipeline

1. **Detection Input**: Receives AprilTag detections with individual tag pose estimates
2. **YTD Algorithm**: Uses yaw angle, tag identity, and distance for localization
3. **Camera Calibration**: Applies intrinsic parameters and distortion correction
4. **Ground-based Optimization**: Specialized for ground-level robot navigation scenarios
5. **Map Registration**: Transforms pose from tag-local coordinates to global map coordinates
6. **Pose Optimization**: Refines pose estimate using geometric constraints

## Parameters

### `camera_parameters_path` (str)

- **Default**: "{project_root}/config/camera_parameters.json"
- **Restart Required**: Yes
- **Description**: Path to camera calibration parameters file containing intrinsic matrix and distortion coefficients.

### `apriltag_map_path` (str)

- **Default**: "{project_root}/config/apriltag_map.fmap"
- **Restart Required**: Yes
- **Description**: Path to AprilTag map file (.fmap) containing known tag positions and orientations in global coordinates.

## Configuration Example

### Pipeline Config Entry

```json
{
    "action_name": "ytd_camera_localization",
    "action_params": {
        "camera_parameters_path": "config/camera_parameters.json",
        "apriltag_map_path": "config/apriltag_map.fmap"
    }
}
```

### Python Usage Example

```python
from src.main_operations.definitions.ytd_camera_localization import YtdCameraLocalizationDefinition
from src.main_operations.definitions.detect_apriltags import DetectApriltagsDefinition
import cv2
import numpy as np

# Initialize AprilTag detector
detector = DetectApriltagsDefinition()

# Initialize YTD pose estimator
localizer = YtdCameraLocalizationDefinition(
    camera_parameters_path="config/camera_parameters.json",
    apriltag_map_path="config/apriltag_map.fmap"
)

frame = cv2.imread("scene_with_tags.jpg")

# Detect tags
detections = detector.run(frame)

# Estimate camera pose using YTD algorithm
camera_pose = localizer.run(detections)

if camera_pose is not None:
    print("Camera position:", camera_pose[:3, 3])
    print("Camera rotation matrix:")
    print(camera_pose[:3, :3])
else:
    print("Pose estimation failed")
```

## Performance Considerations

### Accuracy Optimization

- Specialized for ground-based robot navigation scenarios
- Effective when camera is approximately level with AprilTags
- Good performance with moderate numbers of visible tags
- Robust to varying tag orientations and distances

### Speed Optimization

- Efficient algorithm suitable for real-time applications
- Lower computational overhead than full PnP methods
- Good performance on resource-constrained platforms

### Robustness

- Designed for typical robot navigation geometries
- Works well with tags at various heights and orientations
- Maintains accuracy across different viewing distances

## Tuning Guide

### Camera Orientation

The YTD algorithm performs best when the camera is approximately level. Dynamic camera orientation can be set at runtime:

```python
# Set camera yaw angle (rotation around vertical axis)
localizer.set_attribute("camera_yaw", 0.0)  # radians

# Set camera pitch angle (up/down tilt)
localizer.set_attribute("camera_pitch", 0.1)  # radians
```

### Tag Map Configuration

1. **Tag Placement**: Distribute tags at appropriate heights for ground-based navigation
2. **Coordinate System**: Ensure consistent global coordinate system
3. **Tag Density**: Balance between coverage and processing requirements

## Use Cases

### Ground Robot Navigation

Mobile robot localization using wall or ceiling-mounted AprilTags:

```json
{
    "camera_parameters_path": "config/robot_camera.json",
    "apriltag_map_path": "config/navigation_tags.fmap"
}
```

### Indoor Positioning

Precise indoor positioning for automated guided vehicles:

```json
{
    "camera_parameters_path": "config/agv_camera.json",
    "apriltag_map_path": "config/facility_layout.fmap"
}
```

### Service Robotics

Localization for service robots in structured environments:

```json
{
    "camera_parameters_path": "config/service_bot_camera.json",
    "apriltag_map_path": "config/building_map.fmap"
}
```

## Limitations

1. **Camera Orientation**: Optimized for approximately level camera configurations
2. **Tag Geometry**: Best performance with tags distributed in navigation-appropriate patterns
3. **Distance Range**: Performance may vary with extreme tag distances
4. **Motion Dynamics**: May require camera orientation compensation for dynamic platforms
5. **Calibration Dependency**: Accuracy depends on quality of camera calibration and tag map

## Visualization

This operation returns pose data only and does not provide frame visualization. The `visualize()` method returns `None`.

### Dynamic Attribute Setting

The operation supports runtime adjustment of camera orientation:

```python
# Update camera yaw based on IMU or other sensors
localizer.set_attribute("camera_yaw", current_yaw_radians)

# Adjust for camera tilt
localizer.set_attribute("camera_pitch", current_pitch_radians)
```

## Related Operations

- `PnpCameraLocalizationDefinition`: General-purpose PnP localization
- `FusedCameraLocalizationDefinition`: Combines multiple localization methods
- `DetectApriltagsDefinition`: Provides AprilTag detections for pose estimation

## Files

- **Definition**: `src/main_operations/definitions/ytd_camera_localization.py`
- **Implementation**: `src/main_operations/modules/apriltags/ytd_camera_localization/ytd_localization.py`
- **Config Definition**: `src/main_operations/definitions/config_data/ytd_camera_localization_config_def.json`
- **Pipeline Config Example**: `src/config/pipeline_config.json`
