# Color Threshold Detection Operation

## Overview

The `ColorThresholdDetectionDefinition` is a main pipeline operation that performs color-based object detection with preprocessing. It downscales and letterboxes input frames, converts RGB color ranges to HSV for thresholding, extracts contours, and calculates bounding boxes.

## Operation Type

**Main Operation** - Uses thin-wrapper pattern with implementation in `src/main_operations/modules/object_detection/color_threshold_detection/`

## Category

`det` - Detection operation

## Input/Output

- **Input**: `np.ndarray` (BGR image of any size)
- **Output**: `List[Dict[str, Any]]` with detection information

### Detection Dictionary Format

Each detection contains:

- `bbox`: `[x1, y1, x2, y2]` - Bounding box as percentages (0-1) of letterboxed image dimensions
- `class_id`: `int` - Integer class identifier for the color
- `color_name`: `str` - String name of detected color
- `area`: `float` - Contour area in letterboxed coordinates

## Processing Pipeline

1. **Letterboxing**: Downscales input frame to target size (default 320x320) while maintaining aspect ratio, padding with gray (114)
2. **Noise Reduction**: Applies Gaussian blur for noise reduction (optional)
3. **Color Space Conversion**: Converts BGR to HSV color space
4. **Color Thresholding**: Converts RGB color ranges to HSV and creates binary masks for each configured color range
5. **Mask Cleaning**: Applies morphological operations (opening and closing) to remove noise
6. **Contour Extraction**: Finds contours in each mask
7. **Bounding Box Calculation**: Calculates bounding boxes from contours
8. **Area Filtering**: Filters detections by minimum and maximum area
9. **Coordinate Scaling**: Scales bounding boxes back to original image coordinates
10. **Sorting**: Sorts detections by confidence score (descending)

## Parameters

### `target_size` (int)

- **Default**: 320
- **Range**: 64-1024
- **Restart Required**: Yes
- **Description**: Target size for square letterboxed image. Frame will be resized maintaining aspect ratio and padded to this size.

### `color_ranges` (list)

- **Default**: `null` (uses default red color range)
- **Restart Required**: No
- **Description**: List of color dictionaries for multi-object detection. Each dictionary must contain HSV color ranges.

Each dictionary must contain:

```python
{
    "name": "red",           # String name for the color
    "class_id": 0,           # Integer class identifier
    "lower_hsv": [0, 100, 100],    # Lower HSV bound [H, S, V]
    "upper_hsv": [10, 255, 255]   # Upper HSV bound [H, S, V]
}
```

**HSV Values**:

- H (Hue): 0-179
- S (Saturation): 0-255
- V (Value): 0-255

**Common HSV Color Ranges**:

- Red: `lower_hsv: [0, 100, 100]`, `upper_hsv: [10, 255, 255]`
- Blue: `lower_hsv: [100, 100, 100]`, `upper_hsv: [130, 255, 255]`
- Green: `lower_hsv: [40, 50, 50]`, `upper_hsv: [80, 255, 255]`
- Yellow: `lower_hsv: [20, 100, 100]`, `upper_hsv: [30, 255, 255]`
- Orange: `lower_hsv: [10, 100, 100]`, `upper_hsv: [20, 255, 255]`

### `min_area` (int)

- **Default**: 100
- **Range**: 1-100000
- **Restart Required**: No
- **Description**: Minimum contour area in pixels to consider as valid detection. Smaller contours are filtered out as noise.

### `max_area` (int)

- **Default**: 50000
- **Range**: 100-1000000
- **Restart Required**: No
- **Description**: Maximum contour area in pixels to consider as valid detection. Larger contours are filtered out.

### `blur_kernel_size` (int)

- **Default**: 0
- **Range**: 0-31
- **Restart Required**: No
- **Description**: Gaussian blur kernel size for noise reduction. Must be odd number. Set to 0 to disable blurring.

### `morphology_kernel_size` (int)

- **Default**: 5
- **Range**: 3-31
- **Restart Required**: No
- **Description**: Kernel size for morphological operations (opening and closing) to clean up masks. Must be odd number.

### `morphology_iterations` (int)

- **Default**: 0
- **Range**: 1-10
- **Restart Required**: No
- **Description**: Number of iterations for morphological operations. Higher values provide more aggressive noise removal but may affect detection quality.

## Configuration Example

### Pipeline Config Entry

```json
{
    "action_name": "color_threshold_detection",
    "action_params": {
        "target_size": 320,
        "color_ranges": [
            {
                "name": "red",
                "class_id": 0,
                "lower_hsv": [0, 100, 100],
                "upper_hsv": [10, 255, 255]
            },
            {
                "name": "blue",
                "class_id": 1,
                "lower_hsv": [100, 100, 100],
                "upper_hsv": [130, 255, 255]
            },
            {
                "name": "green",
                "class_id": 2,
                "lower_hsv": [40, 50, 50],
                "upper_hsv": [80, 255, 255]
            }
        ],
        "min_area": 100,
        "max_area": 50000,
        "blur_kernel_size": 0,
        "morphology_kernel_size": 5,
        "morphology_iterations": 0
    }
}
```

### Python Usage Example

```python
from src.main_operations.definitions.color_threshold_detection import ColorThresholdDetectionDefinition
import numpy as np
import cv2

color_ranges = [
    {
        "name": "red",
        "class_id": 0,
        "lower_hsv": [0, 100, 100],
        "upper_hsv": [10, 255, 255]
    },
    {
        "name": "blue",
        "class_id": 1,
        "lower_hsv": [100, 100, 100],
        "upper_hsv": [130, 255, 255]
    }
]

detector = ColorThresholdDetectionDefinition(
    target_size=320,
    color_ranges=color_ranges,
    min_area=100,
    max_area=50000
)

frame = cv2.imread("input.jpg")
detections = detector.run(frame)

for detection in detections:
    print(f"Detected {detection['color_name']} at {detection['bbox']}")
```

## Performance Considerations

### Speed Optimization

- Lower `target_size` for faster processing (e.g., 160 or 240)
- Reduce `morphology_iterations` for faster mask processing
- Set `blur_kernel_size` to 0 to skip blurring
- Limit the number of color ranges

### Accuracy Optimization

- Higher `target_size` for better small object detection (e.g., 640)
- Increase `morphology_iterations` for cleaner masks
- Tune `min_area` and `max_area` based on expected object sizes
- Fine-tune HSV ranges for specific lighting conditions

### Memory Usage

- Proportional to `target_size` squared
- Each color range adds minimal overhead

## Tuning Guide

### Finding RGB Colors and Thresholds

Use this script to find HSV ranges for your specific colors, then manually specify target RGB colors and thresholds for configuration:

```python
import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('HSV Tuner')
cv2.createTrackbar('H_min', 'HSV Tuner', 0, 179, nothing)
cv2.createTrackbar('H_max', 'HSV Tuner', 179, 179, nothing)
cv2.createTrackbar('S_min', 'HSV Tuner', 0, 255, nothing)
cv2.createTrackbar('S_max', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('V_min', 'HSV Tuner', 0, 255, nothing)
cv2.createTrackbar('V_max', 'HSV Tuner', 255, 255, nothing)

frame = cv2.imread('your_image.jpg')
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

while True:
    h_min = cv2.getTrackbarPos('H_min', 'HSV Tuner')
    h_max = cv2.getTrackbarPos('H_max', 'HSV Tuner')
    s_min = cv2.getTrackbarPos('S_min', 'HSV Tuner')
    s_max = cv2.getTrackbarPos('S_max', 'HSV Tuner')
    v_min = cv2.getTrackbarPos('V_min', 'HSV Tuner')
    v_max = cv2.getTrackbarPos('V_max', 'HSV Tuner')

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### Adjusting Parameters

#### Area Thresholds

1. Start with default values (100-50000)
2. Run detection and observe results
3. If too many small noise detections, increase `min_area`
4. If missing large objects, increase `max_area`
5. Adjust based on your specific use case

#### Color Thresholds

1. Start with threshold = 0.3 for most applications
2. If too many false positives, decrease threshold (more restrictive)
3. If missing valid detections, increase threshold (more permissive)
4. Adjust lower_hsv and upper_hsv values to match your specific color conditions

### Morphological Operations

- **Too much noise**: Increase `morphology_iterations` or `morphology_kernel_size`
- **Missing small details**: Decrease `morphology_iterations` or `morphology_kernel_size`
- **Holes in detections**: Increase closing iterations specifically

## Use Cases

### Game Piece Detection

Detect colored game pieces in robotics competitions:

```json
{
    "color_ranges": [
        {
            "name": "red_cone",
            "class_id": 0,
            "target_rgb": [1.0, 0.0, 0.0],
            "threshold": 0.3
        },
        {
            "name": "blue_cube",
            "class_id": 1,
            "target_rgb": [0.0, 0.0, 1.0],
            "threshold": 0.3
        }
    ],
    "min_area": 500,
    "max_area": 30000
}
```

### Traffic Light Detection

Detect traffic light colors:

```json
{
    "color_ranges": [
        {
            "name": "red_light",
            "class_id": 0,
            "target_rgb": [1.0, 0.0, 0.0],
            "threshold": 0.2
        },
        {
            "name": "yellow_light",
            "class_id": 1,
            "target_rgb": [1.0, 1.0, 0.0],
            "threshold": 0.2
        },
        {
            "name": "green_light",
            "class_id": 2,
            "target_rgb": [0.0, 1.0, 0.0],
            "threshold": 0.2
        }
    ],
    "min_area": 200,
    "max_area": 10000
}
```

### Ball Tracking

Track colored balls in sports or robotics:

```json
{
    "color_ranges": [
        {
            "name": "orange_ball",
            "class_id": 0,
            "target_rgb": [1.0, 0.5, 0.0],
            "threshold": 0.3
        }
    ],
    "min_area": 1000,
    "max_area": 50000,
    "blur_kernel_size": 7,
    "morphology_iterations": 3
}
```

## Limitations

1. **Lighting Sensitivity**: RGB-to-HSV thresholding is sensitive to lighting conditions. May require different ranges for different environments.
2. **Overlapping Colors**: Cannot distinguish overlapping objects of the same color.
3. **Complex Shapes**: Works best with convex objects; complex shapes may be split into multiple detections.
4. **Color Ambiguity**: Similar colors (e.g., red and orange) may require careful RGB range tuning.

## Visualization

The operation includes a `visualize()` method that draws bounding boxes and labels on frames:

### Features

- **Color-coded boxes**: Bounding boxes are drawn in the actual detected color (red, blue, green, etc.)
- **Labels**: Shows color name and class ID for each detection
- **Thread-safe**: Uses locks to safely access detection data
- **Automatic color mapping**: Maps common color names to BGR values

### Usage

```python
detector = ColorThresholdDetectionDefinition(...)

# Run detection
detections = detector.run(frame)

# Visualize detections on frame
visualized_frame = detector.visualize(frame.copy())
```

### Color Mapping

The visualizer automatically maps color names to BGR values:

- `red` → (0, 0, 255)
- `blue` → (255, 0, 0)
- `green` → (0, 255, 0)
- `yellow` → (0, 255, 255)
- `orange` → (0, 165, 255)
- `purple` → (255, 0, 255)
- `cyan` → (255, 255, 0)
- `pink` → (203, 192, 255)
- `white` → (255, 255, 255)
- `black` → (0, 0, 0)

Unknown colors default to white.

## Related Operations

- `ObjectDetectionDefinition`: ML-based object detection for more complex scenarios
- `DetectApriltagsDefinition`: Fiducial marker detection for precise localization

## Files

- **Definition**: `src/main_operations/definitions/color_threshold_detection.py`
- **Implementation**: `src/main_operations/modules/object_detection/color_threshold_detection/implementation.py`
- **Config Definition**: `src/main_operations/definitions/config_data/color_threshold_detection_config_def.json`
- **Pipeline Config Example**: `src/config/pipeline_config.json` (see `color_threshold_test` entry)
