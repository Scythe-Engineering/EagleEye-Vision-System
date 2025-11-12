# Position Apriltag Preprocessor Operation

## Overview

The `PositionApriltagPreprocessorDefinition` is a main pipeline operation that enhances AprilTag detection speed by preprocessing camera images with a position-based neural network. It crops input images to areas of interest, reducing computational overhead for subsequent AprilTag detection operations.

## Operation Type

**Main Operation** - Uses implementation in `src/main_operations/modules/apriltags/pre_processing/ai_acceleration/position_apriltag_preprocessor.py`

## Category

`prep` - Preprocessing operation

## Input/Output

- **Input**: `np.ndarray` (BGR image) and optional `Tuple[int, int]` (output_size)
- **Output**: `List[Tuple[np.ndarray, Tuple[int, int]]]` - List of cropped image regions with their coordinates

### Output Format

Each output tuple contains:
- `np.ndarray`: Cropped and processed image region
- `Tuple[int, int]`: (x, y) offset coordinates of the region in the original image

## Processing Pipeline

1. **Neural Network Prediction**: Runs position prediction model to identify potential AprilTag locations
2. **Confidence Filtering**: Filters predictions above confidence threshold
3. **Region Cropping**: Extracts rectangular regions around predicted positions
4. **Padding Application**: Adds padding around detected regions using padding factor
5. **Region Processing**: Prepares cropped regions for downstream processing
6. **Coordinate Tracking**: Maintains offset information for result mapping

## Parameters

### `model_path` (str)

- **Default**: "{project_root}/models/position_predictor/model.pth"
- **Restart Required**: Yes
- **Description**: Path to the trained position predictor model weights file (.pth format).

### `device_id` (str)

- **Default**: "MX3_001"
- **Options**: "CPU", "CUDA", "MX3_001", "CORAL"
- **Restart Required**: Yes
- **Description**: Identifier for the compute device to use for neural network inference.

### `conf_threshold` (float)

- **Default**: 0.5
- **Range**: 0.0-1.0
- **Restart Required**: No
- **Description**: Confidence threshold for position predictions. Higher values reduce false positives but may miss valid detections.

### `padding_factor` (float)

- **Default**: 0.3
- **Range**: 0.0-2.0
- **Restart Required**: No
- **Description**: Factor to pad around detected positions. 0.0 = no padding, 0.5 = 50% padding around the detected region.

## Configuration Example

### Pipeline Config Entry

```json
{
    "action_name": "position_apriltag_preprocessor",
    "action_params": {
        "model_path": "models/position_predictor/model.pth",
        "device_id": "MX3_001",
        "conf_threshold": 0.5,
        "padding_factor": 0.3
    }
}
```

### Python Usage Example

```python
from src.main_operations.definitions.position_apriltag_preprocessor import PositionApriltagPreprocessorDefinition
from src.utils.device_management_utils.compute_pool import ComputePool
import cv2
import numpy as np

compute_pool = ComputePool()

preprocessor = PositionApriltagPreprocessorDefinition(
    model_path="models/position_predictor/model.pth",
    device_id="MX3_001",
    compute_pool=compute_pool,
    conf_threshold=0.5,
    padding_factor=0.3
)

frame = cv2.imread("input.jpg")

# Process frame and get cropped regions
regions = preprocessor.run(frame)

if regions is not None:
    for cropped_image, (offset_x, offset_y) in regions:
        print(f"Cropped region at offset ({offset_x}, {offset_y})")
        print(f"Region shape: {cropped_image.shape}")
```

## Performance Considerations

### Speed Optimization

- Significantly reduces AprilTag detection time by focusing on regions of interest
- Neural network inference optimized for target hardware platforms
- Parallel processing of multiple regions when available

### Accuracy Optimization

- Balance confidence threshold to minimize false positives while maintaining detection rate
- Adjust padding factor based on expected AprilTag sizes and positions
- Model performance depends on training data quality and coverage

### Hardware Acceleration

- Supports multiple compute devices: CPU, CUDA (GPU), MX3 (specialized AI), CORAL (TPU)
- Choose device based on available hardware and performance requirements
- MX3_001 provides best performance for supported platforms

## Tuning Guide

### Confidence Threshold

1. **High Precision**: Increase threshold (e.g., 0.7-0.8) to reduce false detections
2. **High Recall**: Decrease threshold (e.g., 0.3-0.4) to catch more potential regions
3. **Balanced**: Use default 0.5 for most applications

### Padding Factor

1. **Tight Cropping**: Lower values (0.1-0.2) for minimal context
2. **Context Preservation**: Higher values (0.4-0.6) to include surrounding area
3. **Tag Size Dependent**: Increase padding for larger or variable-sized AprilTags

### Model Training

The preprocessor requires a trained neural network model. Key considerations:
- Train on representative images from target environment
- Include various lighting conditions and camera angles
- Balance between detection accuracy and processing speed

## Use Cases

### High-Speed AprilTag Detection

Preprocessing for real-time robot navigation with limited compute resources:

```json
{
    "model_path": "models/nav_position_predictor.pth",
    "device_id": "MX3_001",
    "conf_threshold": 0.6,
    "padding_factor": 0.2
}
```

### Resource-Constrained Systems

Optimizing AprilTag detection on embedded or mobile platforms:

```json
{
    "model_path": "models/mobile_position_predictor.pth",
    "device_id": "CPU",
    "conf_threshold": 0.4,
    "padding_factor": 0.5
}
```

### Variable Environment Detection

Adapting to changing lighting and scene conditions:

```json
{
    "model_path": "models/adaptive_position_predictor.pth",
    "device_id": "CUDA",
    "conf_threshold": 0.5,
    "padding_factor": 0.3
}
```

## Limitations

1. **Model Dependency**: Requires trained neural network model for the specific environment
2. **Training Data**: Performance depends on quality and coverage of training images
3. **False Positives**: May generate regions that don't contain AprilTags
4. **Region Overlap**: Multiple regions may overlap or miss tags in close proximity
5. **Dynamic Environments**: May struggle with significant scene changes not represented in training data

## Visualization

The operation includes a `visualize()` method that highlights detected regions:

### Features

- **Region Highlighting**: Detected areas shown in full brightness on dimmed background
- **Thread-safe**: Uses locks to safely access region data
- **Real-time**: Visualizes the most recent preprocessing results

### Usage

```python
preprocessor = PositionApriltagPreprocessorDefinition(...)

# Run preprocessing
regions = preprocessor.run(frame)

# Visualize detected regions
visualized_frame = preprocessor.visualize(frame.copy())
```

The visualization shows a dimmed version of the original frame with detected regions restored to full brightness, making it easy to see which areas were identified for further processing.

## Related Operations

- `DetectApriltagsDefinition`: Processes the cropped regions for AprilTag detection
- `GridApriltagCnnPreprocessorDefinition`: Alternative preprocessing using grid-based approach
- `TemporalAccelerationPreprocessorRustDefinition`: Segments frames for improved detection

## Files

- **Definition**: `src/main_operations/definitions/position_apriltag_preprocessor.py`
- **Implementation**: `src/main_operations/modules/apriltags/pre_processing/ai_acceleration/position_apriltag_preprocessor.py`
- **Config Definition**: `src/main_operations/definitions/config_data/position_apriltag_preprocessor_config_def.json`
- **Pipeline Config Example**: `src/config/pipeline_config.json`
