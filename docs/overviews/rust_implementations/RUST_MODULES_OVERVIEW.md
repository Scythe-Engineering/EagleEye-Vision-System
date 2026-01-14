# Rust Implementations Overview

## Overview

The EagleEye Vision System includes high-performance Rust implementations for critical vision processing operations. These modules provide significant performance improvements over pure Python implementations through optimized algorithms and efficient memory management.

## Architecture

### Module Structure

```
src/rust_implementations/
├── modules/
│   ├── pose_outlier_filter/    # Pose validation and filtering
│   └── temporal_acceleration/    # Region-of-interest prediction
├── module_template/              # Template for new modules
├── build.py                     # Build system
└── create_module.py              # Module creation utility
```

### Python Integration

All Rust modules are compiled as Python extension modules using PyO3, enabling seamless integration with the existing Python-based pipeline system.

## Available Modules

### Pose Outlier Filter

**Location**: [`src/rust_implementations/modules/pose_outlier_filter/`](../../src/rust_implementations/modules/pose_outlier_filter/)

A high-performance pose outlier filter for pose estimation pipelines. This module maintains a history of accepted poses and uses predictive gating to detect and reject outlier measurements.

#### Key Features

- **Predictive Gating**: Uses constant velocity motion model to predict expected pose locations
- **Adaptive Uncertainty**: Uncertainty grows with consecutive rejections, allowing recovery from temporary tracking loss
- **Angular Validation**: Validates both position and orientation consistency
- **History Management**: Maintains bounded history of accepted poses for velocity estimation
- **Robust Reset Logic**: Automatically resets when consecutive rejections exceed threshold

#### Parameters

- `history_size`: Maximum number of poses to keep in history (default: 20)
- `base_sigma`: Base uncertainty for position predictions in meters (default: 0.1)
- `growth_rate`: Rate at which uncertainty grows with rejections (default: 0.2)
- `gate_k`: Multiplier for uncertainty to create gating threshold (default: 3.0)
- `max_consecutive_rejections`: Threshold for gate relaxation (default: 10)
- `relax_factor`: Factor by which to relax gate when max rejections reached (default: 2.0)
- `angular_gate_threshold`: Max angular difference in radians for acceptance (default: 0.5)
- `velocity_smoothing_alpha`: Smoothing factor for velocity estimates (default: 0.3)
- `full_reset_threshold`: Consecutive rejections before full reset (default: 10)

#### Performance

- O(1) prediction and gating operations
- Bounded memory usage through sliding window history
- Suitable for real-time pose estimation applications
- Optimized for minimal latency in pose validation

#### Usage

```python
from pose_outlier_filter import PoseOutlierFilter

# Create filter with default parameters
filter = PoseOutlierFilter()

# Process pose measurements (4x4 transformation matrices as flat arrays)
pose_flat = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
accepted_pose = filter.run(pose_flat)

if accepted_pose is not None:
    print("Pose accepted")
else:
    print("Pose rejected as outlier")
```

For detailed documentation, see [`pose_outlier_filter/README.md`](../../src/rust_implementations/modules/pose_outlier_filter/README.md).

---

### Temporal Acceleration

**Location**: [`src/rust_implementations/modules/temporal_acceleration/`](../../src/rust_implementations/modules/temporal_acceleration/)

A high-performance implementation for temporal acceleration in AprilTag-based pose estimation. This module uses previous pose information to predict and limit image regions that need processing, significantly reducing computational overhead.

#### Key Features

- **Pose-Based Prediction**: Uses last camera pose to predict AprilTag locations in image space
- **Frustum Culling**: Efficiently discards AprilTags outside camera's field of view
- **Region of Interest Generation**: Creates padded bounding boxes around predicted tag locations
- **Configurable Limits**: Bounds maximum number of regions to prevent excessive computation
- **Depth Filtering**: Rejects tags that are too close to camera to avoid numerical issues

#### Parameters

- `padding_factor`: Factor by which to expand bounding boxes around tags (default: 0.35)
- `max_regions`: Maximum number of regions to return (default: 20)
- `min_region_size_px`: Minimum region size in pixels (default: 16)

#### Performance Benefits

- **Reduced Computation**: Focus processing on predicted regions instead of full frame
- **Scalable**: Region count bounded to prevent performance degradation
- **Memory Efficient**: Minimal state storage (only last pose)
- **Real-time Capable**: Optimized for low-latency operation

#### Usage

```python
from temporal_acceleration import TemporalAcceleration

# Initialize with camera parameters and AprilTag data
accel = TemporalAcceleration(
    camera_matrix=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
    distortion_coefficients=[],
    apriltag_ids=[0, 1, 2],
    apriltag_corners=[...],
    apriltag_centers=[...],
)

# Update with current camera pose (4x4 transformation matrix as flat array)
pose_flat = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
accel.back_propagate_input(pose_flat)

# Process frame to get regions of interest
crops_data, regions = accel.process_frame(width=1920, height=1080)

# regions contains [x, y, width, height] bounding boxes for focused processing
for region in regions:
    x, y, w, h = region
    # Process only this region instead of full frame
```

For detailed documentation, see [`temporal_acceleration/README.md`](../../src/rust_implementations/modules/temporal_acceleration/README.md).

---

## Build System

### Building Modules

The Rust modules are built using the provided build system:

```bash
# Build all modules
python src/rust_implementations/build.py

# The build system:
# 1. Discovers all modules in src/rust_implementations/modules/
# 2. Builds each module using Cargo
# 3. Compiles Python extension modules
# 4. Places compiled modules in appropriate location
```

### Module Discovery

The build system automatically discovers modules by looking for directories containing a `Cargo.toml` file:

```python
# From build.py
def get_modules(self) -> list[Path]:
    """Get all module directories."""
    if not self.modules_dir.exists():
        return []

    return [
        d
        for d in self.modules_dir.iterdir()
        if d.is_dir() and (d / self.CARGO_TOML_FILENAME).exists()
    ]
```

## Creating New Modules

### Module Template

A template is provided at [`src/rust_implementations/module_template/`](../../src/rust_implementations/module_template/) to help create new Rust modules.

### Module Structure

Each module should follow this structure:

```
module_name/
├── Cargo.toml              # Cargo manifest with dependencies
├── README.md               # Module documentation
└── src/
    └── lib.rs             # Main library code with PyO3 bindings
```

### PyO3 Integration

All modules must use PyO3 for Python bindings:

```rust
use pyo3::prelude::*;

#[pyclass]
pub struct MyModule {
    // Module state
}

#[pymethods]
impl MyModule {
    #[new]
    pub fn new() -> Self {
        // Constructor
    }

    pub fn run(&mut self, input: Py<PyAny>) -> PyResult<PyObject> {
        // Main processing method
    }
}
```

## Dependencies

### Core Dependencies

- **PyO3**: Python bindings for Rust
- **NumPy**: For array operations (via ndarray crate)
- **Num-traits**: Numeric traits for generic operations

### Build Dependencies

- **Cargo**: Rust package manager
- **Maturin**: Python extension building (if used)

## Integration with Pipeline System

### Secondary Operation Wrappers

Rust modules are typically wrapped by Python secondary operations:

```python
# Example: pose_outlier_filter_rust.py
from pose_outlier_filter import PoseOutlierFilter
from src.main_operations.definitions.base.base_class import OperationInstance

class PoseOutlierFilterRust(OperationInstance):
    def __init__(self, **kwargs):
        self.filter = PoseOutlierFilter(**kwargs)

    def run(self, pose):
        return self.filter.run(pose)
```

### Configuration

Rust modules are configured through the standard pipeline configuration system:

```json
{
    "action_name": "pose_outlier_filter_rust",
    "action_params": {
        "history_size": 20,
        "base_sigma": 0.1,
        "gate_k": 3.0
    }
}
```

## Performance Considerations

### When to Use Rust Modules

- **Performance-Critical Operations**: Operations called at high frequency (e.g., per-frame)
- **Algorithmic Complexity**: Operations with O(n) or higher complexity
- **Memory-Intensive**: Operations processing large arrays or matrices
- **Numerical Computation**: Operations with heavy mathematical computations

### Benchmarking

Always benchmark Rust modules against Python equivalents:

```python
import time

# Python version
start = time.time()
for _ in range(1000):
    result = python_operation(input)
python_time = time.time() - start

# Rust version
start = time.time()
for _ in range(1000):
    result = rust_operation(input)
rust_time = time.time() - start

print(f"Speedup: {python_time / rust_time:.2f}x")
```

## Development Guidelines

### Code Style

- Follow Rust best practices and idioms
- Use `cargo fmt` for consistent formatting
- Run `cargo clippy` for linting
- Document all public functions and methods

### Testing

- Write unit tests in Rust (`#[cfg(test)]`)
- Write integration tests in Python
- Test edge cases and error conditions
- Benchmark performance

### Documentation

- Provide comprehensive README for each module
- Document all parameters with defaults and ranges
- Include usage examples
- Explain algorithm choices and trade-offs

## Troubleshooting

### Build Issues

**Issue**: Module fails to compile

- Check Rust version compatibility
- Verify all dependencies are installed
- Review Cargo.toml for version conflicts

**Issue**: Python import fails

- Verify module was built successfully
- Check module is in Python path
- Ensure PyO3 version compatibility

### Runtime Issues

**Issue**: Segmentation fault

- Check for unsafe code issues
- Verify array dimensions and types
- Review memory management

**Issue**: Performance degradation

- Profile with `cargo flamegraph`
- Check for unnecessary allocations
- Review algorithm complexity

## Future Enhancements

### Planned Features

- **Additional Modules**: More algorithms implemented in Rust
- **SIMD Optimization**: Vectorized operations for CPU
- **GPU Acceleration**: CUDA/OpenCL support for compatible operations
- **Hot Reloading**: Dynamic module reloading during development
- **Performance Profiling**: Built-in profiling and metrics

---

_Last Updated: January 2025_
