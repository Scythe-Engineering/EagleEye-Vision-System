# Temporal Acceleration

A high-performance Rust implementation for temporal acceleration in AprilTag-based pose estimation. This module uses previous pose information to predict and limit image regions that need processing, significantly reducing computational overhead.

## Overview

The Temporal Acceleration module leverages temporal coherence in camera motion to focus computational resources on relevant image regions. By projecting known AprilTag positions using the last known camera pose, it identifies regions of interest (ROIs) where AprilTags are likely to appear in the current frame.

## Key Features

- **Pose-Based Prediction**: Uses last camera pose to predict AprilTag locations in image space
- **Frustum Culling**: Efficiently discards AprilTags outside the camera's field of view
- **Region of Interest Generation**: Creates padded bounding boxes around predicted tag locations
- **Configurable Limits**: Bounds maximum number of regions to prevent excessive computation
- **Depth Filtering**: Rejects tags that are too close to camera to avoid numerical issues
- **Python Integration**: PyO3 extension module for seamless Python integration

## Architecture

### Core Components

- **`TemporalAcceleration`**: Main processing class containing camera parameters and AprilTag data
- **Pose Tracking**: Stores last known camera pose for temporal prediction
- **AprilTag Database**: Maintains tag IDs, corner positions, and center locations
- **Region Computation**: Projects 3D tag positions to 2D image coordinates
- **Bounding Box Generation**: Creates padded, square regions around predicted tag locations

### Key Parameters

- `padding_factor`: Factor by which to expand bounding boxes around tags (default: 0.35)
- `max_regions`: Maximum number of regions to return (default: 20)
- `min_region_size_px`: Minimum region size in pixels (default: 16)

## Usage

```python
from temporal_acceleration import TemporalAcceleration

# Initialize with camera parameters and AprilTag data
accel = TemporalAcceleration(
    camera_matrix=[fx, 0, cx, 0, fy, cy, 0, 0, 1],  # 3x3 camera matrix
    distortion_coefficients=[],  # Distortion coefficients (empty for now)
    apriltag_ids=[0, 1, 2],  # Tag IDs
    apriltag_corners=[...],  # 4 corners * 3 coords per tag (flattened)
    apriltag_centers=[...],  # Center positions (3 coords per tag, flattened)
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

## Algorithm Details

### Pose Integration

1. Accepts 4x4 transformation matrices representing world-to-camera pose
2. Stores pose for temporal prediction across frames

### Region Prediction

1. Transforms AprilTag world coordinates to camera coordinates using current pose
2. Projects 3D points to 2D image coordinates using camera intrinsics
3. Performs frustum culling to eliminate tags outside field of view
4. Computes bounding boxes around visible tag corners

### Bounding Box Creation

1. Finds axis-aligned bounding box of projected tag corners
2. Converts to square region centered on tag with configurable padding
3. Clamps regions to image boundaries
4. Filters regions below minimum size threshold

### Fallback Behavior

- Returns full frame region when no pose is available
- Ensures at least one region is always returned
- Limits total regions to prevent excessive computation

## Performance Benefits

- **Reduced Computation**: Focus processing on predicted regions instead of full frame
- **Scalable**: Region count bounded to prevent performance degradation
- **Memory Efficient**: Minimal state storage (only last pose)
- **Real-time Capable**: Optimized for low-latency operation

## Dependencies

- `pyo3`: Python bindings for extension module
- `ndarray`: N-dimensional array operations (used for 3D transformations)

## Integration Notes

This module is designed to work with AprilTag detection pipelines where:

- Camera poses are estimated in a world coordinate frame
- AprilTag positions are known in the same world frame
- Frame-by-frame processing allows temporal coherence exploitation

The output regions can be used to crop input frames before AprilTag detection, significantly improving processing speed while maintaining detection accuracy.
