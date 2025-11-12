# Object Detection Operation

## Overview

The `ObjectDetectionDefinition` is a main pipeline operation that performs generic object detection using deep learning models. It supports various model formats including compiled models (.dfp) for specialized hardware and ONNX models for CPU/GPU processing, with automatic fallback to CPU if no model is specified.

## Operation Type

**Main Operation** - Uses implementation in `src/main_operations/modules/object_detection/yolo_detection/implementation.py`

## Category

`det` - Detection operation

## Input/Output

- **Input**: `np.ndarray` (BGR image of any size, uint8)
- **Output**: `List[Dict[str, Any]]` with detection information

### Detection Dictionary Format

Each detection contains:

- `bbox`: `[x1, y1, x2, y2]` - Bounding box as percentages (0-1) of image dimensions
- `score`: `float` - Detection confidence score (0-1)
- `class_id`: `int` - Integer class identifier

## Processing Pipeline

1. **Model Loading**: Loads specified model (.dfp or .onnx) or uses CPU fallback
2. **Input Preprocessing**: Resizes and normalizes input image to model requirements
3. **Inference**: Runs neural network inference on preprocessed input
4. **Post-processing**: Converts raw model outputs to detection format (optional separate model)
5. **Confidence Filtering**: Filters detections below confidence threshold
6. **NMS and Limiting**: Applies non-maximum suppression and limits detection count
7. **Coordinate Scaling**: Scales bounding boxes to percentage coordinates

## Parameters

### `model_path` (str)

- **Default**: null
- **Restart Required**: Yes
- **Description**: Path to device-compatible model file. Supports compiled models (.dfp) with optional post-processing or standalone ONNX models. If null, uses CPU fallback.

### `device_id` (str)

- **Default**: null
- **Restart Required**: Yes
- **Description**: Identifier for compute device from ComputePool. Required when using compiled models.

### `post_processing_model_path` (str)

- **Default**: null
- **Restart Required**: Yes
- **Description**: Optional path to ONNX post-processing model for converting raw compiled model outputs to detections.

### `target_width` (int)

- **Default**: 320
- **Range**: 16-4096
- **Restart Required**: Yes
- **Description**: Target model input width. Image will be resized maintaining aspect ratio.

### `target_height` (int)

- **Default**: 320
- **Range**: 16-4096
- **Restart Required**: Yes
- **Description**: Target model input height. Image will be resized maintaining aspect ratio.

### `conf_threshold` (float)

- **Default**: 0.25
- **Range**: 0.0-1.0
- **Restart Required**: No
- **Description**: Minimum confidence threshold for detections. Lower values return more detections but may include false positives.

### `max_detections` (int)

- **Default**: 100
- **Range**: 1-1000
- **Restart Required**: No
- **Description**: Maximum number of detections to return. Limits processing overhead and result size.

### `is_grayscale` (bool)

- **Default**: false
- **Restart Required**: Yes
- **Description**: Whether the model expects grayscale input (single channel) instead of RGB (three channels).

## Configuration Example

### Pipeline Config Entry

```json
{
    "action_name": "object_detection",
    "action_params": {
        "model_path": "models/yolov8n.onnx",
        "device_id": "cpu",
        "target_width": 640,
        "target_height": 640,
        "conf_threshold": 0.25,
        "max_detections": 100,
        "is_grayscale": false
    }
}
```

### Python Usage Example

```python
from src.main_operations.definitions.object_detection import ObjectDetectionDefinition
from src.utils.device_management_utils.compute_pool import ComputePool
import cv2
import numpy as np

compute_pool = ComputePool()

detector = ObjectDetectionDefinition(
    model_path="models/yolov8n.onnx",
    device_id="cpu",
    compute_pool=compute_pool,
    target_width=640,
    target_height=640,
    conf_threshold=0.25
)

frame = cv2.imread("input.jpg")
detections = detector.run(frame)

for detection in detections:
    print(f"Class {detection['class_id']}: {detection['score']:.2f} confidence")
    print(f"Bounding box: {detection['bbox']}")
```

## Performance Considerations

### Speed Optimization

- Use compiled models (.dfp) with specialized hardware for maximum performance
- Reduce `target_width` and `target_height` for faster inference
- Increase `conf_threshold` to reduce post-processing overhead
- Lower `max_detections` to limit result processing

### Accuracy Optimization

- Higher resolution models (`target_width`, `target_height`) for small object detection
- Lower `conf_threshold` for detecting objects with subtle features
- Use appropriate model architecture for your specific objects
- Ensure proper model training data matches your use case

### Hardware Acceleration

- **Compiled Models (.dfp)**: Optimized for specialized AI hardware (e.g., MX3)
- **ONNX Models**: Portable across CPU, GPU, and specialized hardware
- **CPU Fallback**: Automatic fallback when no model/hardware specified

## Tuning Guide

### Model Selection

#### Pre-trained Models

- **YOLOv8n**: Fast, lightweight, good general-purpose detection
- **YOLOv8s/m/l/x**: Larger models with higher accuracy but slower speed
- **Custom Models**: Train on domain-specific data for specialized applications

#### Model Formats

- **.dfp**: Compiled models for maximum performance on supported hardware
- **.onnx**: Universal format supporting CPU, GPU, and specialized hardware
- **CPU Fallback**: No model file needed, uses built-in CPU implementation

### Threshold Tuning

1. Start with default `conf_threshold: 0.25`
2. Lower threshold if missing valid detections
3. Raise threshold if too many false positives
4. Consider object size and lighting conditions

### Resolution Optimization

- **Small objects**: Increase `target_width`/`target_height` (e.g., 640x640)
- **Large objects**: Decrease resolution for speed (e.g., 320x320)
- **Real-time**: Balance between detection quality and frame rate requirements

## Use Cases

### General Object Detection

Detect common objects in natural scenes:

```json
{
    "model_path": "models/yolov8n.onnx",
    "device_id": "gpu",
    "target_width": 640,
    "target_height": 640,
    "conf_threshold": 0.25,
    "max_detections": 50
}
```

### Specialized Hardware Deployment

High-performance detection on edge devices:

```json
{
    "model_path": "models/custom_model.dfp",
    "device_id": "mx3",
    "post_processing_model_path": "models/post_process.onnx",
    "target_width": 320,
    "target_height": 320,
    "conf_threshold": 0.3
}
```

### Real-time Video Analysis

Low-latency detection for live video streams:

```json
{
    "model_path": "models/yolov8n.onnx",
    "device_id": "gpu",
    "target_width": 416,
    "target_height": 416,
    "conf_threshold": 0.4,
    "max_detections": 20
}
```

## Limitations

1. **Model Dependency**: Requires appropriate trained model for specific object types
2. **Hardware Requirements**: Specialized hardware may be needed for compiled models
3. **Training Data**: Model accuracy depends on quality and relevance of training data
4. **Lighting/Conditions**: Performance varies with lighting, occlusion, and environmental factors
5. **Computational Cost**: Deep learning inference requires significant compute resources

## Visualization

The operation includes a `visualize()` method that draws bounding boxes and labels on frames:

### Features

- **Class-specific colors**: Each object class gets a unique random color
- **Confidence scores**: Displays confidence percentage for each detection
- **Class IDs**: Shows integer class identifiers
- **Thread-safe**: Uses locks to safely access detection data
- **Anti-aliased boxes**: Thick colored outlines for clear visibility

### Color Mapping

Colors are automatically generated for each class ID and persist across frames for consistency.

### Usage

```python
detector = ObjectDetectionDefinition(...)

# Run detection
detections = detector.run(frame)

# Visualize detections on frame
visualized_frame = detector.visualize(frame.copy())
```

## Related Operations

- `ColorThresholdDetectionDefinition`: Color-based detection for specific colored objects
- `DetectApriltagsDefinition`: Fiducial marker detection for precise localization

## Files

- **Definition**: `src/main_operations/definitions/object_detection.py`
- **Implementation**: `src/main_operations/modules/object_detection/yolo_detection/implementation.py`
- **Config Definition**: `src/main_operations/definitions/config_data/object_detection_config_def.json`
- **Pipeline Config Example**: `src/config/pipeline_config.json`
