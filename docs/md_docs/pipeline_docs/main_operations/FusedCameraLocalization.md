# Fused Camera Localization Operation

## Overview

The `FusedCameraLocalizationDefinition` is a main pipeline operation that combines multiple localization methods to provide accurate and robust camera pose estimation using AprilTag detections. It integrates different pose estimation algorithms for improved reliability and precision in challenging conditions.

## Operation Type

**Main Operation** - Uses implementation in `src/main_operations/modules/apriltags/fused_localization.py`

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
2. **Multi-algorithm Fusion**: Combines multiple pose estimation approaches (PnP, YTD, etc.)
3. **Camera Calibration**: Applies intrinsic parameters and distortion correction
4. **Robust Estimation**: Uses complementary methods to improve accuracy and reliability
5. **Map Registration**: Transforms pose from tag-local coordinates to global map coordinates
6. **Fusion Logic**: Intelligently combines results from different estimation methods

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
    "action_name": "fused_camera_localization",
    "action_params": {
        "camera_parameters_path": "config/camera_parameters.json",
        "apriltag_map_path": "config/apriltag_map.fmap"
    }
}
```

### Python Usage Example

```python
from src.main_operations.definitions.fused_camera_localization import FusedCameraLocalizationDefinition
from src.main_operations.definitions.detect_apriltags import DetectApriltagsDefinition
import cv2
import numpy as np

# Initialize AprilTag detector
detector = DetectApriltagsDefinition()

# Initialize fused pose estimator
localizer = FusedCameraLocalizationDefinition(
    camera_parameters_path="config/camera_parameters.json",
    apriltag_map_path="config/apriltag_map.fmap"
)

frame = cv2.imread("scene_with_tags.jpg")

# Detect tags
detections = detector.run(frame)

# Estimate camera pose using fused methods
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

- Combines multiple estimation algorithms for improved robustness
- Better performance in challenging lighting or motion conditions
- More reliable when individual methods might fail
- Maintains accuracy across varying numbers of visible tags

### Speed Optimization

- Computational overhead depends on number of fusion methods used
- May be slower than single-algorithm approaches
- Consider using when reliability is more important than speed

### Robustness

- Better handling of partial occlusions and poor geometry
- Improved performance with varying numbers of visible tags
- More stable pose estimates in dynamic environments

## Tuning Guide

### Algorithm Selection

The fused localization automatically selects and combines appropriate algorithms based on:
- Number of visible AprilTags
- Geometric distribution of tags
- Quality of detections
- Previous pose estimates

### Integration with Other Methods

Can be extended to fuse with additional localization sources:
- IMU data for motion constraints
- Odometry for motion prediction
- Additional camera sensors
- GPS or other global positioning systems

## Use Cases

### Challenging Environments

Robust localization in environments with variable lighting or motion:

```json
{
    "camera_parameters_path": "config/camera_intrinsics.json",
    "apriltag_map_path": "config/facility_map.fmap"
}
```

### High-Precision Applications

Where pose accuracy is critical and some performance overhead is acceptable:

```json
{
    "camera_parameters_path": "config/precision_camera.json",
    "apriltag_map_path": "config/high_accuracy_map.fmap"
}
```

### Mobile Robotics

Robot navigation requiring reliable pose estimates despite dynamic conditions:

```json
{
    "camera_parameters_path": "config/robot_camera.json",
    "apriltag_map_path": "config/navigation_map.fmap"
}
```

## Limitations

1. **Computational Complexity**: Higher computational cost than single-method approaches
2. **Algorithm Dependencies**: Requires multiple estimation methods to be effective
3. **Tag Visibility**: Still requires sufficient visible AprilTags for pose estimation
4. **Calibration Dependency**: Accuracy depends on quality of camera calibration and tag map
5. **Parameter Tuning**: May require tuning of fusion weights and thresholds

## Visualization

This operation returns pose data only and does not provide frame visualization. The `visualize()` method returns `None`.

### Attribute Setting

The operation supports dynamic attribute setting for runtime configuration:

```python
# Set fusion parameters at runtime
localizer.set_attribute("fusion_weight_pnp", 0.7)
localizer.set_attribute("fusion_weight_ytd", 0.3)
```

## Related Operations

- `PnpCameraLocalizationDefinition`: Single-method PnP localization
- `YtdCameraLocalizationDefinition`: YTD-based localization method
- `DetectApriltagsDefinition`: Provides AprilTag detections for pose estimation

## Files

- **Definition**: `src/main_operations/definitions/fused_camera_localization.py`
- **Implementation**: `src/main_operations/modules/apriltags/fused_localization.py`
- **Config Definition**: `src/main_operations/definitions/config_data/fused_camera_localization_config_def.json`
- **Pipeline Config Example**: `src/config/pipeline_config.json`
