# RobotLocalToFieldTransform Operation Overview

## Overview

The `RobotLocalToFieldTransform` operation is a secondary pipeline operation that converts detection positions from the robot's local coordinate frame to global field coordinates. This operation enables robots to express detected object positions in a consistent world reference frame, essential for multi-robot coordination and field-aware autonomous behaviors in FRC competitions.

`GroundPlaneIntersection` does not need this operation because it now consumes a field-relative camera pose and outputs field-relative positions directly.

## Architecture

### Coordinate Frame Transformation System

The operation implements rigid body transformation mathematics to convert between coordinate frames:

1. **Transform Reception**: Receives current robot pose via back-propagation
2. **Position Extraction**: Extracts 3D positions from detection data
3. **Coordinate Transformation**: Applies rotation and translation to convert coordinates
4. **Data Preservation**: Maintains both local and global position information

### Real-Time Synchronization

The operation maintains the latest robot transform through back-propagation, ensuring that coordinate transformations use the most current pose estimate available in the pipeline.

## Key Features

### Precise Coordinate Transformation

- **Rigid Body Transform**: Applies full 3D rotation and translation transformations
- **Real-Time Updates**: Uses latest robot pose for current transformations
- **Data Integrity**: Validates all input data and transformation results
- **Finite Checking**: Ensures all calculated positions are numerically valid

### Dual Coordinate Representation

- **Local Preservation**: Maintains original robot-local positions in `position_robot`
- **Global Conversion**: Provides field coordinates in `position_3d`
- **Backward Compatibility**: Preserves all other detection metadata unchanged
- **Coordinate History**: Enables tracking of position transformations

### Robust Error Handling

- **Transform Validation**: Verifies robot transform matrices are valid 4x4 matrices
- **Position Validation**: Checks detection positions for proper format and finiteness
- **Graceful Degradation**: Passes through detections when transformations fail
- **Type Safety**: Handles various input data types and formats

## Configuration

### Parameters

This operation requires no configuration parameters and operates with default behavior suitable for standard robotics coordinate transformations.

### Initialization

```python
field_transform = RobotLocalToFieldTransform()
```

## Data Flow

### Processing Flow

1. **Detection Input**: Receive list of detection dictionaries with 3D positions
2. **Transform Check**: Verify current robot transform is available
3. **Position Processing**: Extract and validate local positions from each detection
4. **Coordinate Transformation**: Apply robot transform to convert to field coordinates
5. **Result Enhancement**: Add transformed positions and preserve local coordinates

### Processing Steps

```
Input: List[Detection Dicts with position_3d]
       ↓
Check for valid robot transform
       ↓
For each detection:
  Extract local position_3d
  Validate position data
  Apply robot transform (rotation + translation)
  Verify transformed position is finite
       ↓
Create updated detection:
  position_robot = original_local
  position_3d = transformed_field
       ↓
Output: Enhanced detection list
```

## Usage Examples

### Basic Coordinate Transformation

```python
# Initialize coordinate transformer
field_transform = RobotLocalToFieldTransform()

# Example detections in robot-local coordinates
local_detections = [
    {
        "position_3d": [2.0, 0.0, 0.5],  # 2m forward, 0.5m high
        "class_id": 1,
        "confidence": 0.95
    },
    {
        "position_3d": [-1.5, 1.2, 0.0],  # 1.5m left, 1.2m right
        "class_id": 2,
        "confidence": 0.87
    }
]

# Transform to field coordinates (requires robot pose via back-propagation)
field_detections = field_transform.run(local_detections)
# Result: Detections with position_3d in field coords, position_robot preserved
```

### Pipeline Integration

```json
{
  "operations": [
    {
      "type": "primary",
      "name": "apriltag_detection"
    },
    {
      "type": "secondary",
      "name": "camera_local_to_robot_transform"
    },
    {
      "type": "secondary",
      "name": "robot_local_to_field_transform",
      "config": {}
    },
    {
      "type": "secondary",
      "name": "back_propagate",
      "config": {
        "action_name": "robot_pose_output"
      }
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
└── robot_local_to_field_transform.py    # Main operation implementation
```

## Technical Details

### Transformation Mathematics

**Rigid Body Transformation:**
```
field_position = rotation_matrix @ local_position + translation_vector
```

Where:
- `rotation_matrix`: 3x3 rotation from robot transform
- `translation_vector`: 3x1 translation from robot transform
- `local_position`: 3D position in robot coordinate frame
- `field_position`: 3D position in global field coordinate frame

### Coordinate System Conventions

- **Robot Frame**: X forward, Y left, Z up (standard robot convention)
- **Field Frame**: Global coordinate system for FRC competition field
- **Transform Format**: 4x4 homogeneous transformation matrix (world-from-robot)

## Integration Points

### Pipeline Integration

- **Back-Propagation Input**: Receives robot pose from upstream pose operations
- **Position Enhancement**: Adds field coordinates to existing detection data
- **Data Flow**: Passthrough operation that enhances rather than replaces data

### Multi-Robot Coordination

- **Global Reference**: Provides consistent coordinate frame across robots
- **Field Awareness**: Enables field-relative navigation and targeting
- **Shared Knowledge**: Supports multi-robot object tracking and collaboration

## Development Notes

### Transform Synchronization

- **Real-Time Updates**: Robot pose should be updated frequently for accuracy
- **Latency Considerations**: Transform age affects coordinate accuracy
- **Extrapolation**: Consider pose prediction for reduced latency

### Performance Characteristics

- **Computational Load**: Matrix-vector multiplication per detection
- **Memory Usage**: Minimal additional storage for coordinate preservation
- **Thread Safety**: Operation on immutable detection data

## Error Handling

### Validation Mechanisms

- **Transform Validation**: Ensures robot transform is valid 4x4 matrix
- **Position Validation**: Checks detection positions are valid 3D vectors
- **Numerical Stability**: Verifies all transformations produce finite results

### Robustness Features

- **Graceful Handling**: Invalid detections pass through unchanged
- **Error Isolation**: Transformation failures don't affect other detections
- **Type Flexibility**: Handles various numeric input formats

## Future Enhancements

### Planned Features

- **Transform Prediction**: Extrapolate robot pose for reduced latency
- **Uncertainty Propagation**: Track position uncertainty through transformations
- **Multi-Robot Fusion**: Coordinate transformations across robot teams
- **Historical Tracking**: Maintain position history in both coordinate frames
- **Coordinate System Options**: Support for different robotics conventions
- **Transform Validation**: Built-in validation of coordinate system consistency
