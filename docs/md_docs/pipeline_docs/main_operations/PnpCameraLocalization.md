# PnP Camera Localization Operation

## Overview

The `PnpCameraLocalizationDefinition` is a main pipeline operation that calculates camera pose (position and orientation) relative to known AprilTag markers using Perspective-n-Point (PnP) algorithms. It combines camera intrinsics, distortion correction, and fiducial map data to provide accurate 6DOF pose estimation for robot localization.

## Operation Type

**Main Operation** - Uses implementation in `src/main_operations/modules/apriltags/pnp_localization.py`

## Category

`loc` - Localization operation

## Input/Output

- **Input**: `List[Detection]` - AprilTag detection objects from detection operations
- **Output**: `np.ndarray` or `None` - 4x4 transformation matrix representing camera pose in global coordinates

### Pose Matrix Format

The output transformation matrix is a 4x4 numpy array where:
- **Rotation**: Top-left 3x3 submatrix representing camera orientation
- **Translation**: Top-right 3x1 vector representing camera position in meters
- **Homogeneous**: Bottom row is [0, 0, 0, 1] for matrix multiplication compatibility

## Processing Pipeline

1. **Detection Input**: Receives AprilTag detections with corner coordinates
2. **Camera Calibration**: Applies intrinsic parameters and distortion correction
3. **Pose Estimation**: Uses PnP algorithm to solve for camera pose relative to known tag positions
4. **Map Registration**: Transforms pose from tag-local coordinates to global map coordinates
5. **Jump Filtering**: Detects and filters out pose jumps beyond threshold distance
6. **Pose Caching**: Maintains pose history for stability and smoothing

## Parameters

### `camera_parameters_path` (str)

- **Default**: "{project_root}/config/camera_parameters.json"
- **Restart Required**: Yes
- **Description**: Path to camera calibration parameters file containing intrinsic matrix and distortion coefficients.

### `apriltag_map_path` (str)

- **Default**: "{project_root}/config/apriltag_map.fmap"
- **Restart Required**: Yes
- **Description**: Path to AprilTag map file (.fmap) containing known tag positions and orientations in global coordinates.

### `jump_threshold` (float)

- **Default**: 2.0
- **Range**: 0.1-10.0
- **Restart Required**: No
- **Description**: Maximum distance threshold in meters for pose jumps. Sudden pose changes beyond this distance trigger cache clearing to prevent drift.

## Configuration Example

### Pipeline Config Entry

```json
{
    "action_name": "pnp_camera_localization",
    "action_params": {
        "camera_parameters_path": "config/camera_parameters.json",
        "apriltag_map_path": "config/apriltag_map.fmap",
        "jump_threshold": 2.0
    }
}
```

### Python Usage Example

```python
from src.main_operations.definitions.pnp_camera_localization import PnpCameraLocalizationDefinition
from src.main_operations.definitions.detect_apriltags import DetectApriltagsDefinition
import cv2
import numpy as np

# Initialize AprilTag detector
detector = DetectApriltagsDefinition()

# Initialize pose estimator
localizer = PnpCameraLocalizationDefinition(
    camera_parameters_path="config/camera_parameters.json",
    apriltag_map_path="config/apriltag_map.fmap"
)

frame = cv2.imread("scene_with_tags.jpg")

# Detect tags
detections = detector.run(frame)

# Estimate camera pose
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

- Use high-quality camera calibration for accurate intrinsic parameters
- Ensure AprilTag map is precisely surveyed and up-to-date
- Lower `jump_threshold` for more aggressive filtering of pose jumps
- Use multiple visible tags for better pose constraints

### Speed Optimization

- Minimal computational overhead beyond PnP solving
- Performance scales with number of visible AprilTags
- Consider pre-filtering detections to reduce input size

### Robustness

- Requires at least 4 non-coplanar tag points for stable pose estimation
- Performance degrades with fewer visible tags
- Distance and angle to tags affect accuracy

## Tuning Guide

### Camera Calibration

1. **Intrinsic Parameters**: Use proper camera calibration with known focal lengths and principal point
2. **Distortion Correction**: Include radial and tangential distortion coefficients
3. **Calibration Quality**: Ensure low reprojection error (< 1 pixel)

### AprilTag Map Creation

1. **Precise Surveying**: Measure tag positions and orientations with high accuracy
2. **Coordinate System**: Define consistent global coordinate system
3. **Tag Spacing**: Distribute tags for good geometric constraints
4. **Regular Updates**: Recalibrate map if tags are moved or environment changes

### Jump Threshold Tuning

- **Conservative**: Higher threshold (e.g., 5.0m) allows more pose variation
- **Aggressive**: Lower threshold (e.g., 0.5m) filters more pose noise
- **Application Dependent**: Consider robot speed and required precision

## Use Cases

### Robot Navigation

Indoor robot localization using ceiling or wall-mounted AprilTags:

```json
{
    "camera_parameters_path": "config/camera_intrinsics.json",
    "apriltag_map_path": "config/facility_map.fmap",
    "jump_threshold": 1.0
}
```

### AR/VR Tracking

Precise camera tracking for augmented reality applications:

```json
{
    "camera_parameters_path": "config/ar_camera_params.json",
    "apriltag_map_path": "config/tracking_space.fmap",
    "jump_threshold": 0.5
}
```

### Industrial Automation

Automated guided vehicle (AGV) positioning in warehouses:

```json
{
    "camera_parameters_path": "config/industrial_camera.json",
    "apriltag_map_path": "config/warehouse_layout.fmap",
    "jump_threshold": 3.0
}
```

## Limitations

1. **Tag Visibility**: Requires sufficient visible AprilTags for pose estimation
2. **Geometric Constraints**: Needs well-distributed, non-coplanar tags for accuracy
3. **Motion Blur**: Fast camera movement can cause detection and pose errors
4. **Lighting Conditions**: Poor lighting affects tag detection reliability
5. **Calibration Dependency**: Accuracy depends on quality of camera calibration and tag map

## Visualization

This operation returns pose data only and does not provide frame visualization. The `visualize()` method returns `None`.

### Integration with Other Operations

Combine with AprilTag detection visualization:

```python
# Detect and visualize tags
detections = detector.run(frame)
visualized_frame = detector.visualize(frame.copy())

# Estimate pose (no direct visualization)
camera_pose = localizer.run(detections)
```

## Related Operations

- `DetectApriltagsDefinition`: Provides AprilTag detections for pose estimation
- `FusedCameraLocalizationDefinition`: Combines multiple localization sources
- `YtdCameraLocalizationDefinition`: Alternative localization using different sensors

## Files

- **Definition**: `src/main_operations/definitions/pnp_camera_localization.py`
- **Implementation**: `src/main_operations/modules/apriltags/pnp_localization.py`
- **Config Definition**: `src/main_operations/definitions/config_data/pnp_camera_localization_config_def.json`
- **Pipeline Config Example**: `src/config/pipeline_config.json`
