# Temporal Acceleration Preprocessor Rust Operation

## Overview

The `TemporalAccelerationPreprocessorRustDefinition` is a main pipeline operation that uses temporal information and camera motion prediction to accelerate AprilTag detection. It leverages a high-performance Rust implementation to predict regions of interest (ROIs) based on back-propagated camera poses, significantly reducing computational overhead for subsequent detection operations.

## Operation Type

**Main Operation** - Uses Rust implementation via `temporal_acceleration` module

## Category

`prep` - Preprocessing operation

## Input/Output

- **Input**: `np.ndarray` (BGR image frame)
- **Output**: `Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]` - Rectified crops with full-frame transforms and the original frame

### Output Format

Returns a tuple containing:
- `List[Tuple[np.ndarray, np.ndarray]]`: List of `(cropped_image, mapping)` tuples
- `np.ndarray`: Original input frame (unchanged)

Rectified crops use a 3x3 transform that maps detector coordinates back into the original frame. An unrectified fallback crop uses a 2-element `[left, top]` offset instead.

## Processing Pipeline

1. **Pose Back-propagation**: Receives camera pose estimates from previous pipeline iterations
2. **Motion Extrapolation**: Predicts camera movement and projects known AprilTag positions
3. **ROI Prediction**: Calculates projected quadrilaterals for expected AprilTag locations
4. **Region Cropping**: Perspective-aligns predicted regions with configurable padding
5. **Size Filtering**: Applies minimum region size constraints
6. **Region Limiting**: Caps maximum number of returned regions

## Parameters

### `camera_parameters_path` (str)

- **Default**: None (required)
- **Restart Required**: Yes
- **Description**: Path to camera calibration parameters file containing intrinsic matrix and distortion coefficients.

### `apriltag_map_path` (str)

- **Default**: None (required)
- **Restart Required**: Yes
- **Description**: Path to AprilTag map file (.fmap) containing known tag positions and orientations in global coordinates.

### `padding_factor` (float)

- **Default**: 0.65
- **Range**: 0.0-2.0
- **Restart Required**: No
- **Description**: Fractional padding applied around projected tag bounds. Higher values include more context but increase processing area.

### `max_regions` (int)

- **Default**: 10
- **Range**: 1-256
- **Restart Required**: No
- **Description**: Maximum number of ROI regions to return. Limits computational overhead while ensuring coverage.

### `min_region_size_px` (int)

- **Default**: 16
- **Range**: 4-2048
- **Restart Required**: No
- **Description**: Minimum side length for ROI squares in pixels. Regions smaller than this threshold are rejected.

## Configuration Example

### Pipeline Config Entry

```json
{
    "action_name": "temporal_acceleration_preprocessor_rust",
    "action_params": {
        "camera_parameters_path": "config/camera_parameters.json",
        "apriltag_map_path": "config/apriltag_map.fmap",
        "padding_factor": 0.65,
        "max_regions": 10,
        "min_region_size_px": 16
    }
}
```

### Python Usage Example

```python
from src.main_operations.definitions.temporal_acceleration_preprocessor_rust import TemporalAccelerationPreprocessorRustDefinition
import cv2
import numpy as np

preprocessor = TemporalAccelerationPreprocessorRustDefinition(
    camera_bus_id="basic_test",
    apriltag_map_path="files/apriltag_map_path/frc2025r2.json",
    padding_factor=0.35,
    max_regions=20
)

camera_pose = np.eye(4)  # 4x4 identity matrix as example
frame = cv2.imread("input.jpg")

# Generate predicted ROIs using the latest pose from localization
regions, original_frame = preprocessor.run(
    {"frame": frame, "camera_pose": camera_pose}
)

print(f"Generated {len(regions)} predicted regions")
for cropped_image, mapping in regions:
    print(f"Region shape: {cropped_image.shape}, mapping shape: {mapping.shape}")
```

## Performance Considerations

### Speed Optimization

- High-performance Rust implementation for maximum speed
- Temporal prediction reduces unnecessary AprilTag detection attempts
- Configurable region limits prevent excessive computation

### Accuracy Optimization

- Camera pose accuracy directly affects prediction quality
- Appropriate padding balances coverage vs computational cost
- Region size filtering prevents processing of irrelevant small areas

### Temporal Aspects

- Requires stable pose estimates for effective prediction
- Benefits from consistent frame rates and smooth camera motion
- Performance improves with accurate motion models

## Tuning Guide

### Padding Factor

1. **Conservative**: Lower values (0.1-0.3) for precise regions
2. **Robust**: Higher values (0.4-0.7) for motion uncertainty and partial occlusion
3. **Motion-dependent**: Increase with faster camera movement or less precise pose estimation

### Region Limits

1. **Performance Priority**: Lower `max_regions` (5-15) for speed-critical applications
2. **Coverage Priority**: Higher `max_regions` (20-50) for comprehensive scene coverage
3. **Scene Complexity**: Adjust based on expected number of visible AprilTags

### Minimum Region Size

1. **Small Tags**: Lower values (8-16px) for distant or small AprilTags
2. **Large Tags**: Higher values (32-64px) to filter noise and irrelevant small regions
3. **Resolution Dependent**: Scale with camera resolution and expected tag sizes

## Use Cases

### High-Speed Robot Navigation

Temporal prediction for real-time navigation systems:

```json
{
    "camera_parameters_path": "config/nav_camera.json",
    "apriltag_map_path": "config/navigation_tags.fmap",
    "padding_factor": 0.4,
    "max_regions": 15,
    "min_region_size_px": 20
}
```

### Autonomous Vehicle Localization

Predictive preprocessing for self-driving applications:

```json
{
    "camera_parameters_path": "config/vehicle_camera.json",
    "apriltag_map_path": "config/road_tags.fmap",
    "padding_factor": 0.5,
    "max_regions": 25,
    "min_region_size_px": 24
}
```

### AR/VR Tracking Systems

Smooth tracking with motion prediction:

```json
{
    "camera_parameters_path": "config/ar_camera.json",
    "apriltag_map_path": "config/tracking_space.fmap",
    "padding_factor": 0.3,
    "max_regions": 30,
    "min_region_size_px": 16
}
```

## Limitations

1. **Pose Dependency**: Requires accurate camera pose estimates for effective prediction
2. **Motion Assumptions**: Assumes relatively smooth camera motion between frames
3. **Map Accuracy**: Depends on precise AprilTag map calibration
4. **Unexpected Motion**: Struggles with sudden direction changes or accelerations
5. **Occlusion Handling**: May predict regions for occluded but expected tags

## Visualization

The operation includes a `visualize()` method that highlights predicted regions:

### Features

- **Region Highlighting**: Shows predicted ROI locations on dimmed background
- **Multiple Regions**: Displays all predicted regions simultaneously
- **Thread-safe**: Uses locks to safely access region data
- **Real-time**: Visualizes the most recent predictions

### Usage

```python
preprocessor = TemporalAccelerationPreprocessorRustDefinition(...)

# Back-propagate pose and run prediction
preprocessor.back_propagate_input(camera_pose)
regions, frame = preprocessor.run(frame)

# Visualize predicted regions
visualized_frame = preprocessor.visualize(frame.copy())
```

The visualization shows a dimmed version of the frame with predicted regions restored to full brightness, making it easy to see where the temporal prediction expects AprilTags to appear.

## Related Operations

- `DetectApriltagsDefinition`: Processes predicted regions for AprilTag detection
- `PnpCameraLocalizationDefinition`: Provides pose estimates for back-propagation
- `BackPropagateOperation`: Feeds pose data back through the pipeline

## Files

- **Definition**: `src/main_operations/definitions/temporal_acceleration_preprocessor_rust.py`
- **Implementation**: `temporal_acceleration` Rust module (compiled)
- **Config Definition**: `src/main_operations/definitions/config_data/temporal_acceleration_preprocessor_rust_config_def.json`
- **Pipeline Config Example**: `src/config/pipeline_config.json`
