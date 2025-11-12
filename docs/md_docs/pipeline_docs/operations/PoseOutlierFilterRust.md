# Pose Outlier Filter Rust Operation

## Overview

The `PoseOutlierFilterRust` is a secondary pipeline operation that provides high-performance pose outlier filtering using a Rust implementation. It maintains a history of accepted poses and uses constant velocity prediction with uncertainty growth to detect and reject outlier measurements, ensuring robust pose estimation in dynamic environments.

## Operation Type

**Secondary Operation** - High-performance pose filtering utility

## Category

`filter` - Data filtering operation

## Input/Output

- **Input**: `np.ndarray` - 4x4 homogeneous transformation matrix representing a pose
- **Output**: `np.ndarray` or `None` - Accepted pose or None if rejected as outlier

### Processing Behavior

The filter evaluates each pose measurement against predicted values based on motion history, rejecting measurements that fall outside acceptable uncertainty bounds.

## Parameters

### Constructor Parameters

- `history_size` (int): Maximum number of accepted poses to keep in history (default: 20)
- `base_sigma` (float): Base uncertainty for position predictions in meters (default: 0.1)
- `growth_rate` (float): Rate at which uncertainty grows with consecutive rejections (default: 0.2)
- `gate_k` (float): Multiplier for uncertainty to create gating threshold (default: 3.0)
- `max_consecutive_rejections` (int): Max rejections before gate relaxation (default: 10)
- `relax_factor` (float): Factor by which to relax gate when max rejections reached (default: 2.0)
- `angular_gate_threshold` (float): Max angular difference in radians for acceptance (default: 0.5)
- `velocity_smoothing_alpha` (float): Smoothing factor for velocity estimates (0-1) (default: 0.3)
- `full_reset_threshold` (int): Number of consecutive rejections to trigger full filter reset (default: 10)

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
            "action_name": "pose_outlier_filter_rust",
            "action_params": {
                "history_size": 25,
                "base_sigma": 0.08,
                "gate_k": 2.5,
                "max_consecutive_rejections": 8
            }
        }
    ]
}
```

### Python Usage Example

```python
from src.secondary_operations.pose_outlier_filter_rust import PoseOutlierFilterRust
import numpy as np

# Initialize pose filter with conservative settings
pose_filter = PoseOutlierFilterRust(
    history_size=25,        # Keep more history for stability
    base_sigma=0.08,        # Tighter position uncertainty
    gate_k=2.5,            # Moderate gating threshold
    max_consecutive_rejections=8
)

# Example pose from camera localization
camera_pose = np.eye(4)  # 4x4 identity matrix
camera_pose[0, 3] = 1.0  # X position
camera_pose[1, 3] = 0.5  # Y position
camera_pose[2, 3] = 0.0  # Z position

# Filter the pose
filtered_pose = pose_filter.run(camera_pose)

if filtered_pose is not None:
    print("Pose accepted:", filtered_pose[:3, 3])
else:
    print("Pose rejected as outlier")
```

## Performance Considerations

### High-Performance Implementation

- **Rust Backend**: Compiled Rust code provides C-like performance with memory safety
- **Minimal Overhead**: Efficient algorithms for real-time pose filtering
- **Memory Efficient**: Controlled history size prevents unbounded memory growth

### Filtering Algorithm

- **Predictive Gating**: Uses motion prediction to validate new measurements
- **Adaptive Uncertainty**: Uncertainty grows with consecutive rejections
- **Velocity Smoothing**: Exponential smoothing of velocity estimates
- **Angular Validation**: Separate threshold for orientation differences

### Robustness Features

- **Gate Relaxation**: Automatically relaxes thresholds during high-rejection periods
- **Full Reset**: Complete filter reset when consecutive rejections exceed threshold
- **History Management**: Bounded history prevents performance degradation

## Tuning Guide

### History Size

1. **Stable Environments**: Larger history (20-30) for better prediction accuracy
2. **Dynamic Environments**: Smaller history (10-15) for faster adaptation
3. **Memory Constraints**: Balance with available system memory

### Uncertainty Parameters

1. **Precise Localization**: Lower `base_sigma` (0.05-0.1) for tight filtering
2. **Noisy Measurements**: Higher `base_sigma` (0.15-0.3) for more tolerance
3. **Motion-dependent**: Adjust based on expected platform velocity

### Gate Thresholds

1. **Conservative Filtering**: Lower `gate_k` (2.0-2.5) for stricter outlier rejection
2. **Permissive Filtering**: Higher `gate_k` (3.0-4.0) for accepting more measurements
3. **Application Dependent**: Stricter for safety-critical pose estimation

### Rejection Handling

1. **Recovery Speed**: Lower `max_consecutive_rejections` (5-10) for faster recovery
2. **Stability**: Higher values (15-20) prevent premature relaxation
3. **Reset Threshold**: Set `full_reset_threshold` based on expected outlier burst duration

## Use Cases

### Robot Localization Filtering

Filtering noisy pose estimates in dynamic robot navigation:

```json
{
    "history_size": 30,
    "base_sigma": 0.05,
    "gate_k": 2.0,
    "max_consecutive_rejections": 5
}
```

### Camera Pose Stabilization

Smoothing camera localization in shaky or moving platforms:

```json
{
    "history_size": 20,
    "base_sigma": 0.1,
    "gate_k": 3.0,
    "velocity_smoothing_alpha": 0.5
}
```

### Multi-Sensor Fusion

Preprocessing poses before fusion with other localization sources:

```json
{
    "history_size": 25,
    "base_sigma": 0.08,
    "gate_k": 2.5,
    "angular_gate_threshold": 0.3
}
```

## Limitations

1. **Initialization Period**: Requires several valid measurements to establish motion history
2. **Motion Assumptions**: Assumes relatively constant velocity between measurements
3. **Sudden Direction Changes**: May reject valid poses during abrupt maneuvers
4. **Parameter Tuning**: Requires careful tuning for specific application dynamics
5. **Memory Constraints**: History size limited by available system memory

## Visualization

The operation returns pose data only and does not provide frame visualization. The `visualize()` method returns `None`.

### Integration with Monitoring

Consider integrating with monitoring systems to track:
- Acceptance/rejection ratios
- Filter reset events
- Uncertainty growth patterns
- Motion prediction accuracy

## Related Operations

- `PnpCameraLocalizationDefinition`: Provides pose estimates for filtering
- `FusedCameraLocalizationDefinition`: Alternative localization with built-in filtering
- `RobotPoseOutput`: Can consume filtered pose estimates

## Files

- **Definition**: `src/secondary_operations/pose_outlier_filter_rust.py`
- **Implementation**: `pose_outlier_filter` Rust module (compiled)
