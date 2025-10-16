# Pose Outlier Filter

A high-performance Rust implementation of a pose outlier filter for pose estimation pipelines. This module maintains a history of accepted poses and uses predictive gating to detect and reject outlier measurements.

## Overview

The Pose Outlier Filter implements a constant velocity prediction model with adaptive uncertainty growth. It maintains a sliding window of accepted poses and uses Euclidean distance with adaptive thresholds to determine whether new pose measurements are consistent with expected motion.

## Key Features

- **Predictive Gating**: Uses constant velocity motion model to predict expected pose locations
- **Adaptive Uncertainty**: Uncertainty grows with consecutive rejections, allowing recovery from temporary tracking loss
- **Angular Validation**: Validates both position and orientation consistency
- **History Management**: Maintains bounded history of accepted poses for velocity estimation
- **Robust Reset Logic**: Automatically resets when consecutive rejections exceed threshold
- **Python Integration**: PyO3 extension module for seamless Python integration

## Architecture

### Core Components

- **`PoseOutlierFilter`**: Main filter class containing all state and logic
- **History Management**: Sliding window of accepted poses with timestamps
- **Velocity Tracking**: Exponential smoothing of velocity estimates
- **Covariance Estimation**: Rolling window covariance computation for statistical gating
- **State Management**: Tracks consecutive rejections and handles reset scenarios

### Key Parameters

- `history_size`: Maximum number of poses to keep in history (default: 20)
- `base_sigma`: Base uncertainty for position predictions in meters (default: 0.1)
- `growth_rate`: Rate at which uncertainty grows with rejections (default: 0.2)
- `gate_k`: Multiplier for uncertainty to create gating threshold (default: 3.0)
- `max_consecutive_rejections`: Threshold for gate relaxation (default: 10)
- `relax_factor`: Factor by which to relax gate when max rejections reached (default: 2.0)
- `angular_gate_threshold`: Max angular difference in radians for acceptance (default: 0.5)
- `velocity_smoothing_alpha`: Smoothing factor for velocity estimates (default: 0.3)
- `full_reset_threshold`: Consecutive rejections before full reset (default: 10)

## Usage

```python
from pose_outlier_filter import PoseOutlierFilter

# Create filter with default parameters
filter = PoseOutlierFilter()

# Process pose measurements (4x4 transformation matrices as flat arrays)
pose_flat = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # Identity matrix
accepted_pose = filter.run(pose_flat)

if accepted_pose is not None:
    print("Pose accepted")
else:
    print("Pose rejected as outlier")
```

## Algorithm Details

### Prediction Step

1. Uses last accepted pose and velocity to predict current position
2. Applies exponential smoothing to velocity estimates
3. Grows uncertainty based on consecutive rejections

### Gating Step

1. Computes Euclidean distance between predicted and measured positions
2. Calculates angular error using rotation matrix comparison
3. Applies adaptive gating threshold: gate_k × (base_sigma × (1 + growth_rate × consecutive_rejections))
4. Accepts pose if both position and angular errors are below their respective thresholds

### Update Step

1. Updates velocity estimates when pose is accepted
2. Maintains rolling covariance for statistical validation
3. Resets rejection counter and uncertainty

## Dependencies

- `pyo3`: Python bindings
- `ndarray`: High-performance N-dimensional arrays
- `num-traits`: Numeric traits for generic operations

## Performance Characteristics

- O(1) prediction and gating operations
- Bounded memory usage through sliding window history
- Suitable for real-time pose estimation applications
- Optimized for minimal latency in pose validation
