# Grid Apriltag CNN Preprocessor Operation

## Overview

The `GridApriltagCnnPreprocessorDefinition` is a main pipeline operation that enhances AprilTag detection speed by preprocessing camera images with a convolutional neural network. It uses a grid-based approach to identify and preserve regions of interest while eliminating areas unlikely to contain AprilTags.

## Operation Type

**Main Operation** - Uses implementation in `src/main_operations/modules/apriltags/pre_processing/ai_acceleration/grid_apriltag_cnn_preprocessor.py`

## Category

`prep` - Preprocessing operation

## Input/Output

- **Input**: `np.ndarray` (BGR image) and optional `Tuple[int, int]` (output_size)
- **Output**: `np.ndarray` or `None` - Processed frame with non-ROI regions replaced with black pixels

### Output Format

The output frame has the same dimensions as the input, but regions identified as unlikely to contain AprilTags are filled with black pixels, effectively masking them out for downstream processing.

## Processing Pipeline

1. **Grid Division**: Divides input image into a regular grid of regions
2. **CNN Classification**: Runs neural network on each grid cell to predict AprilTag presence
3. **Confidence Filtering**: Filters grid cells based on confidence threshold
4. **Region Masking**: Preserves high-confidence regions, masks low-confidence areas with black
5. **Output Generation**: Returns modified frame with irrelevant areas eliminated
6. **Region Tracking**: Maintains list of preserved regions for visualization

## Parameters

### `model_path` (str)

- **Default**: "{project_root}/models/apriltag_cnn/model.pth"
- **Restart Required**: Yes
- **Description**: Path to the trained CNN model weights file (.pth format).

### `device_id` (str)

- **Default**: "MX3_001"
- **Options**: "CPU", "CUDA", "MX3_001", "CORAL"
- **Restart Required**: Yes
- **Description**: Identifier for the compute device to use for neural network inference.

### `conf_threshold` (float)

- **Default**: 0.15
- **Range**: 0.0-1.0
- **Restart Required**: No
- **Description**: Confidence threshold for grid cell classification. Cells below this threshold are masked out.

## Configuration Example

### Pipeline Config Entry

```json
{
    "action_name": "grid_apriltag_cnn_preprocessor",
    "action_params": {
        "model_path": "models/apriltag_cnn/model.pth",
        "device_id": "MX3_001",
        "conf_threshold": 0.15
    }
}
```

### Python Usage Example

```python
from src.main_operations.definitions.grid_apriltag_cnn_preprocessor import GridApriltagCnnPreprocessorDefinition
from src.utils.device_management_utils.compute_pool import ComputePool
import cv2
import numpy as np

compute_pool = ComputePool()

preprocessor = GridApriltagCnnPreprocessorDefinition(
    model_path="models/apriltag_cnn/model.pth",
    device_id="MX3_001",
    compute_pool=compute_pool,
    conf_threshold=0.15
)

frame = cv2.imread("input.jpg")

# Process frame and get masked output
processed_frame = preprocessor.run(frame)

if processed_frame is not None:
    print(f"Processed frame shape: {processed_frame.shape}")
    # Non-AprilTag regions are now black pixels
```

## Performance Considerations

### Speed Optimization

- Reduces AprilTag detection computational load by eliminating irrelevant image regions
- Grid-based processing allows for efficient parallel classification
- Significantly faster than full-frame AprilTag detection in sparse environments

### Accuracy Optimization

- Lower confidence threshold preserves more potential regions but reduces speed benefits
- Higher threshold eliminates more background but risks missing valid detections
- Model accuracy depends on training data quality and environmental representation

### Hardware Acceleration

- Supports multiple compute devices: CPU, CUDA (GPU), MX3 (specialized AI), CORAL (TPU)
- Optimized for parallel processing of grid cells
- MX3_001 provides best performance for supported platforms

## Tuning Guide

### Confidence Threshold

1. **Conservative**: Lower threshold (0.1-0.2) to preserve more potential regions
2. **Aggressive**: Higher threshold (0.3-0.5) to eliminate more background
3. **Balanced**: Default 0.15 for most applications

### Grid Size Considerations

The grid size is determined by the trained model architecture. Consider:
- **Fine Grid**: More precise region selection but higher computational cost
- **Coarse Grid**: Faster processing but may miss small or partially occluded tags
- **Adaptive**: Choose based on expected AprilTag sizes and scene complexity

### Model Training

Key requirements for the CNN model:
- Train on grid cells containing AprilTags at various scales and orientations
- Include diverse background regions for robust classification
- Balance between false positive and false negative rates

## Use Cases

### Sparse Environment Detection

Optimizing detection in environments with AprilTags scattered among complex backgrounds:

```json
{
    "model_path": "models/sparse_apriltag_cnn.pth",
    "device_id": "MX3_001",
    "conf_threshold": 0.2
}
```

### Real-time Robot Navigation

Preprocessing for high-speed navigation systems:

```json
{
    "model_path": "models/nav_cnn_model.pth",
    "device_id": "CUDA",
    "conf_threshold": 0.15
}
```

### Embedded System Optimization

Reducing computational load on resource-constrained platforms:

```json
{
    "model_path": "models/embedded_apriltag_cnn.pth",
    "device_id": "CPU",
    "conf_threshold": 0.25
}
```

## Limitations

1. **Model Dependency**: Requires trained CNN model specific to the target environment
2. **Grid Constraints**: Performance depends on chosen grid resolution
3. **Training Data**: Accuracy limited by quality and diversity of training images
4. **Partial Occlusion**: May mask regions containing partially visible AprilTags
5. **Dynamic Scenes**: Struggles with significant scene changes not represented in training

## Visualization

The operation includes a `visualize()` method that shows preserved vs eliminated regions:

### Features

- **Region Preservation**: Shows only the grid cells identified as containing potential AprilTags
- **Black Masking**: Eliminated regions appear as black areas
- **Thread-safe**: Uses locks to safely access region data
- **Real-time**: Visualizes the most recent preprocessing results

### Usage

```python
preprocessor = GridApriltagCnnPreprocessorDefinition(...)

# Run preprocessing
processed_frame = preprocessor.run(frame)

# Visualize preserved regions
visualized_frame = preprocessor.visualize(frame.copy())
```

The visualization displays only the regions that passed the confidence threshold, with eliminated areas appearing black, making it clear which parts of the image are being processed for AprilTag detection.

## Related Operations

- `DetectApriltagsDefinition`: Processes the preprocessed frame for AprilTag detection
- `PositionApriltagPreprocessorDefinition`: Alternative position-based preprocessing
- `TemporalAccelerationPreprocessorRustDefinition`: Segments frames using temporal information

## Files

- **Definition**: `src/main_operations/definitions/grid_apriltag_cnn_preprocessor.py`
- **Implementation**: `src/main_operations/modules/apriltags/pre_processing/ai_acceleration/grid_apriltag_cnn_preprocessor.py`
- **Config Definition**: `src/main_operations/definitions/config_data/grid_apriltag_cnn_preprocessor_config_def.json`
- **Pipeline Config Example**: `src/config/pipeline_config.json`
