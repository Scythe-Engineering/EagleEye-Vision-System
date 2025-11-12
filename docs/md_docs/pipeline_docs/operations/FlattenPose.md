# Flatten Pose Operation

## Overview

The `FlattenPose` is a secondary pipeline operation that converts 3D pose estimates to 2D by removing height information and rotational components other than yaw. This operation is useful for ground-based robots that operate primarily in 2D space, simplifying pose representation for navigation and control systems.

## Operation Type

**Secondary Operation** - Pose transformation utility

## Category

`transform` - Data transformation operation

## Input/Output

- **Input**: `np.ndarray` - 4x4 homogeneous transformation matrix (3D pose)
- **Output**: `np.ndarray` - 4x4 transformation matrix flattened to 2D

### Processing Behavior

Extracts yaw rotation from the 3D pose and creates a 2D pose matrix with zero height and no roll/pitch rotations.

## Parameters

This operation has no configurable parameters.

## Configuration Example

### Pipeline Integration

```json
{
    "object_detection_pipeline": [
        {
            "action_name": "pnp_camera_localization",
            "action_params": {
                "camera_parameters_path": "config/camera.json",
                "apriltag_map_path": "config/tags.fmap"
            }
        },
        {
            "action_name": "flatten_pose",
            "action_params": {}
        }
    ]
}
```

### Python Usage Example

```python
from src.secondary_operations.flatten_pose import FlattenPose
import numpy as np

flatten_pose = FlattenPose()

# Example 3D pose from camera localization
pose_3d = np.array([
    [0.866, -0.500, 0.000, 2.5],   # Rotation with yaw ≈ 30°
    [0.500,  0.866, 0.000, 1.8],   # and translation (2.5, 1.8, 0.3)
    [0.000,  0.000, 1.000, 0.3],
    [0.000,  0.000, 0.000, 1.0]
])

# Flatten to 2D
pose_2d = flatten_pose.run(pose_3d)

print("2D Position:", pose_2d[:2, 3])  # (2.5, 1.8)
print("Height:", pose_2d[2, 3])        # 0.0 (flattened)
print("Yaw preserved from original rotation")
```

## Mathematical Transformation

### Input Pose Structure

The input is a 4x4 homogeneous transformation matrix:
```
[R11, R12, R13, Tx]
[R21, R22, R23, Ty]
[R31, R32, R33, Tz]
[0,   0,   0,   1 ]
```

Where:
- **R**: 3x3 rotation matrix
- **T**: 3x1 translation vector (Tx, Ty, Tz)

### Output Pose Structure

The output is a flattened 2D pose matrix:
```
[cos(θ), -sin(θ), 0, Tx]
[sin(θ),  cos(θ), 0, Ty]
[0,       0,       1, 0 ]
[0,       0,       0, 1 ]
```

Where:
- **θ**: Yaw angle extracted from input rotation
- **Tx, Ty**: Original X, Y translation (preserved)
- **Tz**: Set to 0 (flattened to ground plane)

## Performance Considerations

### Computational Efficiency

- **Minimal Operations**: Only trigonometric functions and array assignments
- **No Dependencies**: Pure numpy operations with no external libraries
- **Constant Time**: Processing time independent of pose complexity

### Memory Usage

- **Input Copying**: Creates a copy of the input pose matrix
- **No Additional Storage**: No internal state or history maintained
- **Fixed Size**: Always produces 4x4 output regardless of input

### Numerical Stability

- **Angle Extraction**: Uses atan2 for robust angle calculation
- **Trigonometric Precision**: Standard numpy implementations
- **Matrix Orthogonality**: Maintains valid rotation matrix properties

## Use Cases

### Ground Robot Navigation

Converting 3D camera localization to 2D for differential drive robots:

```json
{
    "action_name": "flatten_pose",
    "action_params": {}
}
```

### 2D Mapping and Localization

Preparing pose data for 2D SLAM or occupancy grid mapping:

```json
{
    "action_name": "flatten_pose",
    "action_params": {}
}
```

### Control System Integration

Simplifying pose representation for PID controllers and motion planning:

```json
{
    "action_name": "flatten_pose",
    "action_params": {}
}
```

## Implementation Details

### Yaw Extraction

```python
yaw_angle = np.arctan2(flattened_pose[1, 0], flattened_pose[0, 0])
```

Uses the rotation matrix elements to extract the yaw angle using atan2 for proper quadrant handling.

### Pose Construction

```python
flattened_pose[0, 0] = cos_yaw
flattened_pose[0, 1] = -sin_yaw
flattened_pose[0, 2] = 0.0
flattened_pose[1, 0] = sin_yaw
flattened_pose[1, 1] = cos_yaw
flattened_pose[1, 2] = 0.0
flattened_pose[2, 0] = 0.0
flattened_pose[2, 1] = 0.0
flattened_pose[2, 2] = 1.0
```

Constructs a valid 2D rotation matrix with preserved yaw and zero roll/pitch.

## Coordinate System Conventions

### Input Assumptions

- **Z-Up**: Assumes Z-axis points up (standard robotics convention)
- **Right-Handed**: Standard right-handed coordinate system
- **Camera Frame**: Poses in camera or robot coordinate frame

### Output Guarantees

- **2D Motion**: Only X, Y translation and Z-axis rotation preserved
- **Ground Plane**: Z position set to 0 (on ground plane)
- **Valid Transform**: Maintains properties of homogeneous transformation matrix

## Limitations

1. **Yaw-Only Rotation**: Discards roll and pitch information permanently
2. **Ground Plane Assumption**: Assumes robot operates on flat surface at Z=0
3. **No Height Information**: Cannot recover original Z position after flattening
4. **Rotation Matrix Dependency**: Requires valid rotation matrix in input pose
5. **No Configuration**: Fixed transformation behavior with no parameters

## Visualization

The operation does not provide frame visualization as it processes pose transformation data only. The `visualize()` method returns `None`.

### Integration with Visualization

Flattened poses can be visualized using the web interface or other pose visualization operations that accept 2D pose data.

## Related Operations

- **3D Pose Sources**: `PnpCameraLocalizationDefinition`, `FusedCameraLocalizationDefinition`
- **2D Consumers**: Operations expecting 2D pose data for navigation/control
- **Pose Processing**: Other pose transformation and filtering operations

## Files

- **Definition**: `src/secondary_operations/flatten_pose.py`
