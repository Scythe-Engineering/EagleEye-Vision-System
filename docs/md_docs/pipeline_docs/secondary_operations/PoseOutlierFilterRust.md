# PoseOutlierFilterRust Operation Overview

## Overview

The `PoseOutlierFilterRust` operation is a high-performance secondary pipeline operation that provides pose outlier detection and filtering using a Rust-based implementation. This operation maintains a history of accepted poses and uses predictive filtering with uncertainty modeling to detect and reject anomalous pose measurements, ensuring robust pose estimation in dynamic robotics environments.

## Architecture

### High-Performance Filtering System

The operation implements advanced pose filtering through:

1. **Rust Backend**: High-performance Rust implementation for computationally intensive operations
2. **Python Wrapper**: Clean Python interface with automatic data conversion
3. **Predictive Gating**: Constant velocity prediction with adaptive uncertainty modeling
4. **History Management**: Circular buffer of accepted poses for velocity estimation

### Adaptive Filtering Strategy

The filter dynamically adjusts its acceptance criteria based on system behavior, relaxing constraints during periods of high uncertainty while maintaining strict validation during normal operation.

## Key Features

### Advanced Outlier Detection

- **Predictive Filtering**: Uses constant velocity model to predict expected pose
- **Uncertainty Modeling**: Adaptive uncertainty growth based on consecutive rejections
- **Multi-Criteria Gating**: Combines position and angular difference thresholds
- **Velocity Smoothing**: Exponential smoothing of velocity estimates for stability

### Performance Optimization

- **Rust Implementation**: Compiled Rust backend for maximum performance
- **Memory Efficiency**: Fixed-size circular buffers for pose history
- **Zero-Copy Operations**: Minimized data copying between Python and Rust
- **SIMD Operations**: Vectorized mathematical operations in Rust

### Robust State Management

- **Automatic Reset**: Full filter reset after extended rejection periods
- **Gate Relaxation**: Progressive threshold relaxation during uncertainty periods
- **History Validation**: Continuous validation of stored pose history
- **Thread Safety**: Designed for use in multi-threaded pipeline environments

## Configuration

### Core Parameters

- **history_size**: Maximum accepted poses in history buffer (default: 20)
- **base_sigma**: Base position uncertainty in meters (default: 0.1)
- **growth_rate**: Uncertainty growth rate per rejection (default: 0.2)
- **gate_k**: Uncertainty multiplier for gating threshold (default: 3.0)

### Advanced Parameters

- **max_consecutive_rejections**: Rejections before gate relaxation (default: 10)
- **relax_factor**: Gate relaxation multiplier (default: 2.0)
- **angular_gate_threshold**: Maximum angular difference in radians (default: 0.5)
- **velocity_smoothing_alpha**: Velocity smoothing factor 0-1 (default: 0.3)
- **full_reset_threshold**: Consecutive rejections for full reset (default: 10)

### Configuration Example

```python
pose_filter = PoseOutlierFilterRust(
    history_size=25,
    base_sigma=0.05,
    growth_rate=0.15,
    gate_k=2.5,
    max_consecutive_rejections=8,
    angular_gate_threshold=0.3
)
```

## Data Flow

### Processing Flow

1. **Pose Reception**: Accept 4x4 homogeneous transformation matrix
2. **Prediction Generation**: Predict expected pose using velocity model
3. **Uncertainty Calculation**: Compute adaptive uncertainty bounds
4. **Gating Test**: Compare measurement against prediction gates
5. **Acceptance Decision**: Accept/reject based on position and angular criteria

### Processing Steps

```
Input: 4x4 Pose Matrix
       ↓
Generate pose prediction from history
       ↓
Calculate adaptive uncertainty bounds
       ↓
Test measurement against position gate
       ↓
Test measurement against angular gate
       ↓
If accepted: Add to history, return pose
If rejected: Return None, increase uncertainty
```

## Usage Examples

### Basic Pose Filtering

```python
# Initialize high-precision pose filter
pose_filter = PoseOutlierFilterRust(
    base_sigma=0.02,        # 2cm base uncertainty
    history_size=30,        # 30 pose history
    gate_k=2.0              # 2-sigma gating
)

# Filter pose measurement
filtered_pose = pose_filter.run(raw_pose_measurement)
if filtered_pose is not None:
    # Use accepted pose for robot control
    robot_controller.update_pose(filtered_pose)
else:
    # Handle rejected measurement (outlier)
    logging.warning("Pose measurement rejected as outlier")
```

### Pipeline Integration

```json
{
  "operations": [
    {
      "type": "primary",
      "name": "apriltag_detection"
    },
    {
      "type": "secondary",
      "name": "pose_estimation"
    },
    {
      "type": "secondary",
      "name": "pose_outlier_filter_rust",
      "config": {
        "base_sigma": 0.03,
        "history_size": 25,
        "gate_k": 2.5
      }
    },
    {
      "type": "secondary",
      "name": "robot_pose_output"
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
└── pose_outlier_filter_rust.py    # Python wrapper implementation

pose_outlier_filter/               # Rust crate directory
├── src/
│   └── lib.rs                    # Rust implementation
├── Cargo.toml                    # Rust dependencies
└── build.rs                      # Build configuration
```

## Technical Details

### Filtering Algorithm

**Prediction Model:**
- Constant velocity assumption between measurements
- Exponential smoothing of velocity estimates
- Adaptive uncertainty growth with consecutive rejections

**Gating Criteria:**
```
position_gate = k * (base_sigma + growth_rate * consecutive_rejections)
angular_gate = angular_gate_threshold

acceptance = (position_error < position_gate) AND (angular_error < angular_gate)
```

### Rust Integration

**Data Conversion:**
- Numpy arrays converted to flat f64 vectors for Rust
- Results converted back to 4x4 numpy arrays
- Automatic memory management across language boundary

### Performance Characteristics

- **Computational Complexity**: O(history_size) for velocity calculation
- **Memory Usage**: Fixed allocation based on history_size parameter
- **Language Boundary**: Minimal overhead for Python-Rust data transfer

## Integration Points

### Pipeline Integration

- **Pose Validation**: Filters pose estimates before robot control usage
- **Outlier Handling**: Provides None returns for rejected measurements
- **State Continuity**: Maintains filtering state across pipeline executions

### Robotics Applications

- **Localization Robustness**: Prevents pose estimate corruption from outliers
- **Control System Stability**: Provides reliable pose data for feedback control
- **Sensor Fusion**: Complements other pose estimation methods

## Development Notes

### Build Requirements

- **Rust Toolchain**: Rust compiler and Cargo build system
- **PyO3**: Python bindings for Rust-Python interop
- **Maturin**: Build system for Python packages with Rust extensions

### Performance Considerations

- **History Size**: Larger history improves velocity estimation but increases computation
- **Uncertainty Parameters**: Tuned based on sensor characteristics and motion dynamics
- **Reset Thresholds**: Configured based on expected outlier frequency

## Error Handling

### Filter State Management

- **Initialization**: Proper setup validation and parameter checking
- **Boundary Conditions**: Handling of empty history and initialization periods
- **Numerical Stability**: Validation of all mathematical operations

### Robustness Features

- **Automatic Recovery**: Filter reset and relaxation mechanisms
- **Parameter Validation**: Runtime checking of configuration parameters
- **Exception Safety**: Graceful handling of computational errors

## Future Enhancements

### Planned Features

- **Multi-Model Filtering**: Support for different motion models (constant acceleration, etc.)
- **Adaptive Parameters**: Automatic parameter tuning based on performance metrics
- **Sensor Fusion**: Integration with IMU and odometry data
- **Performance Monitoring**: Built-in statistics and health monitoring
- **Configuration Profiles**: Preset configurations for different robotics applications
- **Visualization**: Real-time plotting of filter state and uncertainty bounds
