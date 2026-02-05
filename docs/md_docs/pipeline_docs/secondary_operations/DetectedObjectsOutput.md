# DetectedObjectsOutput Operation Overview

## Overview

The `DetectedObjectsOutput` operation is a secondary pipeline operation that forwards validated 3D detection data to the EagleEye web interface for real-time visualization and monitoring. This operation acts as a bridge between the computer vision pipeline and the user interface, providing filtered and validated detection data for 3D visualization.

## Architecture

### Data Validation Pipeline

The operation implements a comprehensive validation system that:

1. **Position Validation**: Ensures 3D position data is valid and finite
2. **Payload Construction**: Builds clean detection payloads for web interface consumption
3. **Change Detection**: Prevents redundant updates by comparing detection signatures
4. **Web Interface Integration**: Sends validated data to the EagleEye web server

### Change Detection Mechanism

The operation uses content-based signatures to avoid sending duplicate detection data, reducing network traffic and web interface update frequency while ensuring real-time responsiveness.

## Key Features

### Data Validation

- **Position Verification**: Validates 3D coordinates are finite numeric values
- **Type Safety**: Ensures all position components are proper numeric types
- **Dimensionality Check**: Confirms 3D position vectors have correct dimensions

### Payload Optimization

- **Selective Field Extraction**: Only includes relevant fields for visualization
- **Data Type Normalization**: Converts coordinates to consistent float representation
- **Optional Field Handling**: Gracefully handles missing detection metadata

### Performance Optimization

- **Change Detection**: Prevents redundant web interface updates
- **Minimal Processing**: Lightweight validation without heavy computation
- **Memory Efficient**: No data duplication or transformation overhead

## Configuration

### Parameters and Dependencies

- **web_interface**: EagleEyeInterface dependency injected automatically when `web_interface` is defined in the constructor. This dependency does not need to be provided via `action_params` because the pipeline inserts it before creation.

### Configuration Example

```python
detected_output = DetectedObjectsOutput(
    web_interface=eagle_eye_web_interface
)
```

## Data Flow

### Input Processing

1. **Null Check**: Handle cases where no detections are available
2. **Detection Validation**: Process each detection dictionary individually
3. **Payload Construction**: Build validated payloads for valid detections
4. **Signature Computation**: Generate content signature for change detection
5. **Update Decision**: Compare with last signature to determine if update needed

### Processing Steps

```
Input: List[Detection Dicts] or None
       ↓
Validate input is not None
       ↓
For each detection:
  Validate 3D position data
  Build clean payload
  Collect valid payloads
       ↓
Compute content signature
       ↓
Compare with last signature
       ↓
Send to web interface (if changed)
       ↓
Output: Original detections unchanged
```

## Usage Examples

### Basic Web Interface Output

```python
# Initialize with web interface
output_op = DetectedObjectsOutput(web_interface)

# Example detections with 3D positions
detections = [
    {
        "position_3d": [1.5, 2.0, 0.0],
        "class_id": 1,
        "class_name": "robot",
        "confidence": 0.95
    },
    {
        "position_3d": [-0.8, 1.2, 0.3],
        "class_id": 2,
        "class_name": "target",
        "confidence": 0.87
    }
]

# Send to web interface
result = output_op.run(detections)
# Web interface receives validated 3D data for visualization
```

### Pipeline Integration

```json
{
  "operations": [
    {
      "type": "primary",
      "name": "object_detection_3d"
    },
    {
      "type": "secondary",
      "name": "detected_objects_output",
      "config": {}
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
└── detected_objects_output.py    # Main operation implementation
```

## Technical Details

### Validation Logic

**Position Validation:**
```python
def _is_valid_position(self, position):
    return (
        isinstance(position, Iterable) and
        len(position) == 3 and
        all(isinstance(coord, (int, float)) and np.isfinite(coord)
            for coord in position)
    )
```

**Payload Construction:**
```python
payload = {
    "position_3d": [float(coord) for coord in position],
    "class_id": detection.get("class_id"),
    "class_name": detection.get("class_name"),
    "confidence": detection.get("confidence")
}
```

### Change Detection

**Signature Computation:**
- Combines position coordinates, class identifiers, and confidence values
- Creates hashable tuple for efficient comparison
- Only includes validated detections in signature

## Integration Points

### Web Interface Integration

- **Real-Time Updates**: Provides live detection data for 3D visualization
- **Data Filtering**: Sends only validated, relevant information
- **Performance Monitoring**: Supports web interface debugging and monitoring

### Pipeline Integration

- **Passthrough Operation**: Returns input detections unchanged for chaining
- **Validation Layer**: Acts as quality gate before web interface consumption
- **Change Filtering**: Reduces update frequency while maintaining responsiveness

## Development Notes

### Data Format Assumptions

- **Detection Structure**: Expects dictionaries with position_3d and optional metadata
- **Coordinate System**: Assumes standard 3D coordinate conventions
- **Field Naming**: Compatible with common computer vision detection formats

### Performance Characteristics

- **Validation Overhead**: Minimal computational cost for data validation
- **Memory Usage**: Temporary storage for payload construction
- **Network Efficiency**: Change detection prevents unnecessary transmissions

## Error Handling

### Validation Failures

- **Invalid Positions**: Detections with invalid 3D positions are filtered out
- **Missing Data**: Gracefully handles incomplete detection dictionaries
- **Type Errors**: Robust handling of unexpected data types

### Robustness Features

- **Null Safety**: Proper handling of None inputs and missing fields
- **Type Conversion**: Safe numeric type conversions with validation
- **Exception Prevention**: Validation prevents runtime errors from malformed data

## Future Enhancements

### Planned Features

- **Data Compression**: Binary serialization for reduced network bandwidth
- **Batch Updates**: Support for efficient batch detection transmission
- **Quality Metrics**: Additional validation and quality scoring
- **Custom Filtering**: Configurable detection filtering rules
- **Historical Tracking**: Detection history and trajectory visualization support
