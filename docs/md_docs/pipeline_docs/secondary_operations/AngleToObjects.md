# AngleToObjects Operation Overview

## Overview

The `AngleToObjects` operation is a secondary pipeline operation that calculates horizontal viewing angles to detected objects from color threshold detection results. This operation converts 2D bounding box coordinates into angular measurements relative to the camera center, enabling robots to determine the directional location of detected objects for navigation and targeting.

## Architecture

### Angle Calculation Process

The operation implements pinhole camera model mathematics to:

1. **Bounding Box Processing**: Extracts center coordinates from detection bounding boxes
2. **Coordinate Normalization**: Converts pixel coordinates to normalized camera coordinates
3. **Angular Conversion**: Uses camera field-of-view to calculate horizontal angles
4. **Result Sorting**: Orders detections by area for priority-based processing

### Pinhole Camera Model

The operation uses geometric camera principles to convert from 2D image coordinates to 3D viewing angles, assuming a rectilinear lens projection model with known horizontal field of view.

## Key Features

### Precise Angle Calculation

- **Field-of-View Integration**: Uses actual camera FOV for accurate angle computation
- **Center-Based Calculation**: Computes angles from object centers for targeting precision
- **Bilateral Angle Range**: Supports full 360-degree horizontal angle calculations

### Detection Processing

- **Area-Based Sorting**: Prioritizes larger detections for more significant targets
- **Metadata Preservation**: Maintains all original detection information
- **Robust Coordinate Handling**: Handles edge cases in bounding box coordinates

### Real-Time Performance

- **Vectorized Operations**: Efficient numpy-based angle calculations
- **Minimal Overhead**: Lightweight processing suitable for real-time vision pipelines
- **Memory Efficient**: No additional data structures beyond output enhancement

## Configuration

### Required Parameters

- **camera_fov_degrees**: Horizontal field of view in degrees (default: 60.0)

### Configuration Example

```python
angle_calculator = AngleToObjects(
    camera_fov_degrees=75.0  # Wide-angle camera
)
```

## Data Flow

### Input Processing

1. **Detection Iteration**: Process each detection in the input list
2. **Bounding Box Extraction**: Get normalized coordinates from detection data
3. **Center Calculation**: Compute horizontal center of each bounding box
4. **Coordinate Transformation**: Convert to camera-centered coordinate system
5. **Angle Computation**: Apply pinhole camera model mathematics
6. **Result Enhancement**: Add angle information to detection dictionaries

### Processing Steps

```
Input: List[Detection Dicts]
       ↓
For each detection:
  Extract bbox coordinates
  Calculate center X position
  Convert to normalized centered coords (-1 to 1)
  Apply camera FOV transformation
  Compute horizontal angle
       ↓
Sort by area (largest first)
       ↓
Output: Enhanced detection list with angles
```

## Usage Examples

### Basic Angle Calculation

```python
# Initialize with camera specifications
angle_op = AngleToObjects(camera_fov_degrees=60.0)

# Example detections from color thresholding
detections = [
    {
        "bbox": [0.2, 0.3, 0.4, 0.6],  # Left side object
        "class_id": 1,
        "color_name": "red",
        "area": 150.0
    },
    {
        "bbox": [0.6, 0.2, 0.8, 0.5],  # Right side object
        "class_id": 2,
        "color_name": "blue",
        "area": 200.0
    }
]

# Calculate angles
angled_objects = angle_op.run(detections)
# Result: Objects with angle_degrees, angle_radians fields
```

### Pipeline Integration

```json
{
  "operations": [
    {
      "type": "primary",
      "name": "color_threshold"
    },
    {
      "type": "secondary",
      "name": "angle_to_objects",
      "config": {
        "camera_fov_degrees": 75.0
      }
    },
    {
      "type": "secondary",
      "name": "publish_to_networktables"
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
└── angle_to_objects.py    # Main operation implementation
```

## Technical Details

### Mathematical Operations

**Coordinate Normalization:**
```
center_x_norm = (bbox[0] + bbox[2]) / 2.0
x_norm_centered = 2.0 * (center_x_norm - 0.5)  # Range: [-1, 1]
```

**Angle Calculation:**
```
horizontal_angle_rad = arctan(x_norm_centered * tan(fov_rad / 2.0))
horizontal_angle_deg = rad2deg(horizontal_angle_rad)
```

### Camera Model Assumptions

- **Rectilinear Projection**: Standard pinhole camera model
- **Horizontal FOV**: Symmetric field of view around optical center
- **Principal Point**: Assumes optical center at image center (0.5, 0.5)

## Integration Points

### Pipeline Integration

- **Detection Source**: Requires color threshold or similar detection operations
- **Angle Consumers**: Provides targeting data for autonomous navigation systems
- **Data Enhancement**: Adds angular information without removing existing data

### Robotics Applications

- **Target Tracking**: Enables robots to point towards detected objects
- **Navigation Planning**: Supports angle-based path planning algorithms
- **Sensor Fusion**: Complements other positioning systems with vision-based angles

## Development Notes

### FOV Calibration

- **Camera Matching**: FOV parameter must match actual camera specifications
- **Calibration Procedures**: Use known targets at measured angles for validation
- **Dynamic Adjustment**: Consider runtime FOV updates for zoom cameras

### Performance Characteristics

- **Linear Complexity**: O(n) processing time for n detections
- **Memory Scaling**: Output size scales with input detection count
- **Numerical Stability**: Clamping prevents angle calculation edge cases

## Error Handling

### Input Validation

- **Detection Format**: Assumes standard detection dictionary structure
- **Bounding Box Validity**: Handles malformed or out-of-range coordinates
- **Missing Fields**: Graceful handling of optional detection fields

### Robustness Features

- **Coordinate Clamping**: Prevents invalid angle calculations
- **Default Values**: Provides fallback values for missing area information
- **Type Safety**: Maintains consistent data types in output

## Future Enhancements

### Planned Features

- **Vertical Angles**: Support for elevation angle calculations
- **Distance Estimation**: Integration with known object sizes for range finding
- **Multi-Camera Support**: Handling for stereo camera configurations
- **Angular Filtering**: Outlier detection and filtering for angle measurements
- **Coordinate System Options**: Support for different angle reference frames
