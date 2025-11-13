# ExtractPose Operation Overview

## Overview

The `ExtractPose` operation is a secondary pipeline operation that extracts 2D pose data (position and rotation) from a 4x4 transformation matrix. This operation converts complex 3D transformation matrices into simple 2D pose representations suitable for robot navigation and control systems.

## Architecture

### Input Processing

The operation accepts a 4x4 transformation matrix representing a 3D pose and extracts:

- **Translation**: X and Y position coordinates from the matrix translation components
- **Rotation**: Yaw angle (rotation around Z-axis) extracted from the 2x2 rotation submatrix

### Output Format

Returns a dictionary containing:
- `x`: X-coordinate position (float)
- `y`: Y-coordinate position (float)
- `rotation`: Rotation angle in radians (float)

## Key Features

### Robust Error Handling

- **None Input Handling**: Returns None when input pose is None (pose estimation failure)
- **Shape Validation**: Validates that input is a proper 4x4 transformation matrix
- **Type Safety**: Ensures all outputs are proper float values

### 2D Pose Extraction

- **Translation Extraction**: Directly extracts X and Y coordinates from matrix translation column
- **Rotation Calculation**: Uses arctan2 for stable angle extraction from rotation matrix
- **Coordinate System**: Preserves standard robotics coordinate conventions

### Real-Time Performance

- **Minimal Computation**: Simple matrix element access and trigonometric operations
- **Memory Efficient**: No additional data structures or copies beyond output dictionary

## Configuration

### Parameters

This operation requires no configuration parameters and operates with default behavior suitable for most robotics applications.

### Initialization

```python
extract_pose_op = ExtractPose()
```

## Data Flow

### Input Processing

1. **Validation**: Check if input pose matrix is None
2. **Shape Check**: Verify matrix is 4x4 transformation matrix
3. **Translation Extraction**: Extract X, Y coordinates from translation column
4. **Rotation Extraction**: Calculate yaw angle from rotation submatrix

### Processing Steps

```
Input: 4x4 transformation matrix or None
       ↓
Validation & Shape Check
       ↓
Extract translation (x, y)
       ↓
Extract rotation angle
       ↓
Output: {"x": float, "y": float, "rotation": float} or None
```

## Usage Examples

### Basic Pose Extraction

```python
# Initialize operation
extract_pose = ExtractPose()

# Example transformation matrix
pose_matrix = np.array([
    [0.866, -0.5, 0, 1.5],   # 30-degree rotation
    [0.5,   0.866, 0, 2.0],  # translation (1.5, 2.0)
    [0,     0,     1, 0],
    [0,     0,     0, 1]
])

# Extract 2D pose
pose_2d = extract_pose.run(pose_matrix)
# Result: {"x": 1.5, "y": 2.0, "rotation": 0.5236}  # ~30 degrees
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
      "name": "extract_pose",
      "config": {}
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
├── extract_pose.py                 # Main operation implementation
└── config_data/
    └── extract_pose_config_def.json # Configuration definition
```

## Technical Details

### Mathematical Operations

**Translation Extraction:**
```
x = pose[0, 3]
y = pose[1, 3]
```

**Rotation Extraction:**
```
rotation_matrix = pose[:2, :2]
rotation = atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
```

### Coordinate System Assumptions

- **X-Axis**: Forward direction (robot heading)
- **Y-Axis**: Lateral direction (left/right)
- **Z-Axis**: Up direction (height)
- **Rotation**: Counter-clockwise positive (standard mathematical convention)

## Integration Points

### Pipeline Integration

- **Input Source**: Typically receives pose matrices from AprilTag detection operations
- **Output Usage**: Provides 2D pose data for navigation, control systems, and NetworkTables publishing
- **Data Flow**: Pure transformation operation - no side effects or external dependencies

### Error Propagation

- **Failure Handling**: Returns None when pose estimation fails, allowing upstream error handling
- **Type Consistency**: Maintains consistent output types regardless of input variations

## Development Notes

### Extending Functionality

The operation could be extended to support:
- **3D Pose Extraction**: Full 6DOF pose extraction with roll/pitch angles
- **Multiple Poses**: Batch processing of multiple transformation matrices
- **Coordinate System Conversion**: Support for different robotics coordinate conventions

### Performance Considerations

- **Computational Complexity**: O(1) - constant time operations
- **Memory Usage**: Minimal - only creates small output dictionary
- **Thread Safety**: Stateless operation, safe for concurrent use

## Error Handling

### Validation Errors

- **Shape Mismatch**: Raises ValueError for non-4x4 input matrices
- **Type Issues**: Input validation ensures proper numpy array types

### Recovery Mechanisms

- **Graceful Degradation**: Returns None for invalid inputs rather than crashing
- **Consistent Output**: Always returns either valid pose dictionary or None

## Future Enhancements

### Planned Features

- **Configuration Options**: Support for different coordinate system conventions
- **Multiple Output Formats**: Support for different pose representation formats
- **Validation Modes**: Configurable strictness for input validation
