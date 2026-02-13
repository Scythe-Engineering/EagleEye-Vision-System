# Pose Fusion

## Overview

The `pose_fusion` operation combines multiple 4x4 transformation matrix pose estimates into a single consensus pose using weighted averaging with outlier rejection. This is useful for multi-camera pipelines where each camera provides an independent pose estimate and you want to merge them into a single, more accurate result.

## Category

`proc` (Processing)

## Algorithm

The pose fusion algorithm operates in three stages:

### 1. Input Validation
- Filters out invalid pose inputs (None, wrong shape, non-finite values)
- Requires at least one valid pose to proceed

### 2. Outlier Rejection (when 4+ poses available)
- Computes median translation and rotation across all poses
- Calculates distance of each pose from the median
- Rejects poses exceeding the `outlier_threshold`
- Uses median-based approach to avoid outliers affecting the reference point

### 3. Weighted Averaging
- Computes cluster center from remaining inlier poses
- Assigns weights inversely proportional to distance from center
- Performs weighted average of:
  - **Translation**: Direct weighted average of position vectors
  - **Rotation**: Quaternion-based weighted average for mathematically correct rotation blending

## Distance Metric

Pose distance is computed as:
```
distance = translation_distance + (rotation_distance × rotation_weight)
```

Where:
- `translation_distance` is Euclidean distance between positions (in meters)
- `rotation_distance` is angular distance between orientations (in radians)
- `rotation_weight` scales rotation contribution relative to translation

## Input Nodes

### Dynamic Input Group
- **Prefix**: `pose`
- **Min inputs**: 1
- **Max inputs**: unlimited
- **Format**: Each input should be a 4x4 numpy array (transformation matrix) or None

The operation accepts inputs in two formats:
1. **Single pose**: Direct 4x4 transformation matrix
2. **Multiple poses**: Dictionary with keys like `pose_0`, `pose_1`, `pose_2`, etc.

Example multi-input:
```python
{
    "pose_0": camera1_pose,  # 4x4 matrix
    "pose_1": camera2_pose,  # 4x4 matrix
    "pose_2": camera3_pose,  # 4x4 matrix
}
```

## Output Nodes

- **fused_pose**: Single 4x4 transformation matrix representing the consensus pose, or None if no valid inputs

## Parameters

### outlier_threshold
- **Type**: float
- **Default**: 1.0
- **Description**: Threshold on the combined distance metric for outlier rejection. The comparison is: `translation_distance (m) + rotation_weight × rotation_distance (rad) ≤ outlier_threshold`. Since this combines meters and weighted radians, the effective units depend on the `rotation_weight` parameter. Poses exceeding this composite threshold are rejected when 4+ inputs are available. Reference: see `rotation_weight` parameter for understanding how rotation contributes to the metric.
- **Tuning**: Decrease for stricter outlier filtering, increase to be more permissive

### rotation_weight
- **Type**: float
- **Default**: 0.5
- **Description**: Weight factor for rotation distance relative to translation distance. Controls how much rotation differences contribute to outlier detection.
- **Tuning**:
  - Increase if rotation accuracy is more important than position
  - Decrease if position accuracy is more important than orientation

## Use Cases

1. **Multi-camera pose fusion**: Combine pose estimates from multiple cameras viewing the same scene
2. **Temporal smoothing**: Fuse current pose with recent historical poses (with appropriate weighting)
3. **Sensor fusion**: Merge poses from different sensor modalities (vision, odometry, IMU-derived)

## Example Configuration

```json
{
    "action_name": "pose_fusion.py",
    "action_params": {
        "outlier_threshold": 0.8,
        "rotation_weight": 0.5
    },
    "position": {"x": 800, "y": 300},
    "uuid": "op-fusion-001",
    "connections": [
        {
            "from_uuid": "op-cam1-pose",
            "from_port": "camera_pose",
            "to_uuid": "op-fusion-001",
            "to_port": "pose_0",
            "data_type": "pose",
            "is_default": false
        },
        {
            "from_uuid": "op-cam2-pose",
            "from_port": "camera_pose",
            "to_uuid": "op-fusion-001",
            "to_port": "pose_1",
            "data_type": "pose",
            "is_default": false
        },
        {
            "from_uuid": "op-fusion-001",
            "from_port": "fused_pose",
            "to_uuid": "op-next-step",
            "to_port": "pose",
            "data_type": "pose",
            "is_default": false
        }
    ]
}
```

## Implementation Details

### Rotation Averaging
The operation uses quaternion-based rotation averaging to ensure mathematically correct blending of orientations. Direct averaging of rotation matrices would produce invalid (non-orthogonal) results. The quaternion approach:

1. Converts each rotation matrix to quaternion representation
2. Ensures all quaternions are in the same hemisphere (corrects sign ambiguity)
3. Computes weighted average in quaternion space
4. Normalizes and converts back to rotation matrix

### Robustness
- **Median-based outlier detection**: Uses median instead of mean for reference point to avoid outliers skewing the threshold
- **Fallback behavior**: If all poses are rejected as outliers, returns all poses as a best-effort estimate rather than None. This ensures downstream operations receive a pose (albeit potentially unreliable) when no consensus can be reached, allowing the pipeline to continue with degraded quality rather than failing. Consumers of the PoseFusion output should implement their own reliability checks if critical decisions depend on pose quality.
- **Single-pose optimization**: When only one valid pose is available, returns it directly without averaging overhead

## Performance Considerations

- **Complexity**: O(n²) where n is number of poses (due to pairwise distance calculations)
- **Suitable for**: Up to ~20 pose inputs per frame
- **Optimization tip**: Pre-filter obviously invalid poses before fusion to reduce computation

## Dependencies

- numpy (rotation matrix, translation operations)
- No external dependencies beyond numpy
