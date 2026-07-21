# Detect Apriltags Operation

## Overview

The `DetectApriltagsDefinition` is a main pipeline operation that detects AprilTag fiducial markers in camera images. It uses the pupil-apriltags library to detect and decode various AprilTag families, extracting tag IDs, corner positions, and pose information for robot localization and navigation.

## Operation Type

**Main Operation** - Uses thin-wrapper pattern with implementation in `src/main_operations/modules/apriltags/apriltag_detector.py`

## Category

`det` - Detection operation

## Input/Output

- **Input**: `np.ndarray` (BGR image) or `tuple` (segments, full_frame) from temporal acceleration preprocessor
- **Output**: `List[Detection]` containing detected AprilTag information

### Detection Dictionary Format

Each detection is a `Detection` object containing:

- `tag_id`: Integer ID of the detected AprilTag
- `corners`: 4x2 numpy array of corner coordinates [(x,y), (x,y), ...]
- `center`: 2-element array of tag center coordinates [x, y]
- `tag_family`: String name of the AprilTag family
- `decision_margin`: Quality metric for detection confidence

## Processing Pipeline

1. **Input Processing**: Accepts either direct image input or segmented input from temporal acceleration preprocessor
2. **AprilTag Detection**: Uses pupil-apriltags library to detect markers in the specified family
3. **Parameter Optimization**: Applies quad decimation, Gaussian blur, edge refinement, and decode sharpening
4. **Multi-threading**: Parallel processing for improved performance

## Parameters

### `families` (str)

- **Default**: "tag36h11"
- **Options**: "tag16h5", "tag25h9", "tag36h11", "tagCircle21h7", "tagCircle49h12", "tagCustom48h12", "tagStandard41h12", "tagStandard52h13"
- **Restart Required**: No
- **Description**: AprilTag family to detect. Different families offer different tag counts and robustness.

### `nthreads` (int)

- **Default**: 1
- **Range**: 1-16
- **Restart Required**: No
- **Description**: Number of threads to use for detection. Higher values improve performance on multi-core systems.

### `quad_decimate` (float)

- **Default**: 2.0
- **Range**: 1.0-10.0
- **Restart Required**: No
- **Description**: Detection of quads can be done on a lower-resolution image, improving speed at a cost of pose accuracy and slight decrease in detection rate.

### `quad_sigma` (float)

- **Default**: 0.0
- **Range**: 0.0-5.0
- **Restart Required**: No
- **Description**: Gaussian blur standard deviation in pixels applied to segmented image for quad detection. Higher values reduce noise but may affect detection.

### `refine_edges` (int)

- **Default**: 1
- **Options**: 0, 1
- **Restart Required**: No
- **Description**: When non-zero, the edges of each quad are adjusted to "snap to" strong gradients nearby, improving accuracy.

### `decode_sharpening` (float)

- **Default**: 0.25
- **Range**: 0.0-1.0
- **Restart Required**: No
- **Description**: How much sharpening should be done to decoded images. Improves decoding reliability for blurry or low-contrast tags.

## Configuration Example

### Pipeline Config Entry

```json
{
    "action_name": "detect_apriltags",
    "action_params": {
        "families": "tag36h11",
        "nthreads": 2,
        "quad_decimate": 1.0,
        "quad_sigma": 0.0,
        "refine_edges": 1,
        "decode_sharpening": 0.25
    }
}
```

### Python Usage Example

```python
from src.main_operations.definitions.detect_apriltags import DetectApriltagsDefinition
import cv2
import numpy as np

detector = DetectApriltagsDefinition(
    families="tag36h11",
    nthreads=2,
    quad_decimate=1.0
)

frame = cv2.imread("scene_with_tags.jpg")
detections = detector.run(frame)

for detection in detections:
    print(f"Detected tag ID: {detection.tag_id}")
    print(f"Corners: {detection.corners}")
    if hasattr(detection, 'pose_t'):
        print(f"Translation: {detection.pose_t}")
```

## Performance Considerations

### Speed Optimization

- Lower `quad_decimate` values (e.g., 1.0) for maximum accuracy but slower performance
- Increase `nthreads` on multi-core systems for parallel processing
- Set `quad_sigma` to 0.0 to skip Gaussian blur
- Use `tag16h5` family for fastest detection (but fewer available tags)

### Accuracy Optimization

- Use `quad_decimate: 1.0` for maximum pose accuracy
- Enable `refine_edges: 1` for better corner precision
- Increase `decode_sharpening` for low-contrast environments
- Choose appropriate tag family based on required tag count vs. detection speed

### Memory Usage

- Minimal memory overhead beyond input image
- Multi-threading increases memory usage proportionally to thread count
- Detection results stored temporarily for visualization

## Tuning Guide

### Tag Family Selection

- **tag16h5**: Fastest detection, 16h5 error correction, 30 tags
- **tag25h9**: Good balance of speed and robustness, 25h9 error correction, 35 tags
- **tag36h11**: Most robust, 36h11 error correction, 2320 tags (recommended for most applications)
- **tagCircle21h7/tagCircle49h12**: Circular tags for better rotation invariance

### Environment-Specific Tuning

#### Low Light Conditions

- Decrease `quad_decimate` for better detection at distance
- Increase `decode_sharpening` to improve tag decoding
- Consider larger tag families for better error correction

#### High Motion Blur

- Decrease `quad_decimate` for more stable detection
- Enable `refine_edges` for better corner accuracy
- Increase `quad_sigma` slightly to reduce noise sensitivity

#### Close Range Detection

- Increase `quad_decimate` for faster processing
- Use smaller tag families for speed
- Consider `quad_sigma > 0` to reduce false detections from texture

## Use Cases

### Robot Localization

Precise indoor positioning using AprilTag landmarks:

```json
{
    "families": "tag36h11",
    "nthreads": 4,
    "quad_decimate": 1.0,
    "refine_edges": 1,
    "decode_sharpening": 0.25
}
```

### Augmented Reality

Marker tracking for AR applications:

```json
{
    "families": "tag25h9",
    "nthreads": 2,
    "quad_decimate": 1.5,
    "quad_sigma": 0.5,
    "refine_edges": 1
}
```

### Industrial Automation

Reliable fiducial detection in manufacturing environments:

```json
{
    "families": "tag36h11",
    "nthreads": 1,
    "quad_decimate": 1.0,
    "refine_edges": 1,
    "decode_sharpening": 0.5
}
```

## Timing

When this operation runs in a pipeline from `DeviceInput`, the pipeline runtime preserves the input frame's timing metadata on its detection-list output. The `Detection` objects themselves do not contain a capture timestamp; the timestamp is carried by the pipeline wrapper and can reach PnP and NetworkTables publishing.

## Limitations

1. **Lighting Sensitivity**: Requires adequate lighting and contrast for reliable detection
2. **Motion Blur**: Fast-moving tags may not be detected reliably
3. **Perspective Distortion**: Extreme viewing angles can reduce detection rate
4. **Occlusion**: Partial occlusion of tags reduces detection confidence
5. **Tag Size**: Very small tags require high-resolution cameras and careful tuning

## Visualization

The operation includes a `visualize()` method that draws detected AprilTags on frames:

### Features

- **Bounding boxes**: Green polygons drawn around detected tag corners
- **Tag IDs**: White text labels showing the tag ID at the center
- **Thread-safe**: Uses locks to safely access detection data
- **Real-time**: Visualizes the most recent detection results

### Usage

```python
detector = DetectApriltagsDefinition(...)

# Run detection
detections = detector.run(frame)

# Visualize detections on frame
visualized_frame = detector.visualize(frame.copy())
```

## Related Operations

- `PnpCameraLocalizationDefinition`: Uses AprilTag detections for 3D camera pose estimation
- `PositionApriltagPreprocessorDefinition`: Preprocesses AprilTag detections for position tracking
- `TemporalAccelerationPreprocessorRustDefinition`: Segments frames for improved AprilTag detection

## Files

- **Definition**: `src/main_operations/definitions/detect_apriltags.py`
- **Implementation**: `src/main_operations/modules/apriltags/apriltag_detector.py`
- **Config Definition**: `src/main_operations/definitions/config_data/detect_apriltags_config_def.json`
- **Pipeline Config Example**: `src/config/pipeline_config.json`
