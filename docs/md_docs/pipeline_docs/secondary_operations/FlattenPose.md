# FlattenPose Operation Overview

## Overview

The `FlattenPose` operation is a secondary pipeline operation that converts 3D pose matrices to 2D by removing the Z position component and eliminating roll/pitch rotations while preserving yaw rotation. This operation is essential for robotics applications that operate in 2D planes while maintaining heading information.

## Architecture

### Pose Transformation Process

The operation performs a geometric transformation that:

1. **Z-Coordinate Removal**: Sets the Z translation component to 0
2. **Rotation Flattening**: Eliminates roll (X-axis) and pitch (Y-axis) rotations
3. **Yaw Preservation**: Maintains the yaw (Z-axis) rotation for heading information

### Mathematical Approach

The flattening process reconstructs the rotation matrix to contain only pure yaw rotation, effectively projecting the 3D pose onto the XY plane while preserving directional orientation.

## Key Features

### Geometric Transformations

- **Z-Position Zeroing**: Removes height information for 2D navigation
- **Rotation Normalization**: Eliminates unwanted rotational degrees of freedom
- **Heading Preservation**: Maintains critical directional information

### Coordinate System Handling

- **2D Projection**: Converts 3D poses to 2D coordinate systems
- **Rotation Matrix Reconstruction**: Creates proper 2D rotation matrices from 3D inputs
- **Matrix Structure Preservation**: Maintains 4x4 homogeneous transformation format

### Real-Time Performance

- **In-Place Operations**: Efficient matrix modifications without full recreation
- **Minimal Computation**: Trigonometric operations only for rotation reconstruction
- **Memory Efficient**: Returns modified copy of input matrix

## Configuration

### Parameters

This operation requires no configuration parameters and operates with mathematical defaults suitable for standard robotics coordinate systems.

### Initialization

```python
flatten_pose_op = FlattenPose()
```

## Data Flow

### Input Processing

1. **Matrix Copy**: Creates working copy to avoid modifying original data
2. **Z-Translation Zeroing**: Sets Z position component to 0
3. **Yaw Extraction**: Calculates current yaw angle from rotation matrix
4. **Rotation Reconstruction**: Builds pure 2D rotation matrix from yaw angle

### Processing Steps

```
Input: 4x4 transformation matrix
       ↓
Create working copy
       ↓
Zero Z translation
       ↓
Extract yaw angle from rotation
       ↓
Reconstruct 2D rotation matrix
       ↓
Output: Flattened 4x4 matrix
```

## Usage Examples

### Basic Pose Flattening

```python
# Initialize operation
flatten_pose = FlattenPose()

# Example 3D pose with roll, pitch, yaw
pose_3d = np.array([
    [0.5, -0.7, 0.5, 1.0],   # Complex 3D rotation
    [0.7, 0.5, -0.5, 2.0],   # with translation
    [0.5, 0.5, 0.7, 0.5],    # Z height = 0.5
    [0, 0, 0, 1]
])

# Flatten to 2D
pose_2d = flatten_pose.run(pose_3d)
# Result: Z translation = 0, only yaw rotation preserved
```

### Pipeline Integration

```json
{
  "operations": [
    {
      "type": "primary",
      "name": "pose_estimation"
    },
    {
      "type": "secondary",
      "name": "flatten_pose",
      "config": {}
    },
    {
      "type": "secondary",
      "name": "extract_pose"
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
└── flatten_pose.py    # Main operation implementation
```

## Technical Details

### Mathematical Operations

**Z Translation Zeroing:**
```
flattened_pose[2, 3] = 0.0
```

**Yaw Angle Extraction:**
```
yaw_angle = atan2(pose[1, 0], pose[0, 0])
```

**2D Rotation Matrix Reconstruction:**
```
cos_yaw = cos(yaw_angle)
sin_yaw = sin(yaw_angle)

rotation_matrix = [
    [cos_yaw, -sin_yaw, 0],
    [sin_yaw,  cos_yaw, 0],
    [0,        0,       1]
]
```

### Coordinate System Assumptions

- **Right-Hand Rule**: Standard 3D coordinate system conventions
- **Rotation Order**: Z-Y-X Euler angle convention (yaw-pitch-roll)
- **Forward Direction**: X-axis represents forward motion

## Integration Points

### Pipeline Integration

- **Input Source**: Receives pose matrices from 3D pose estimation operations
- **Output Usage**: Provides 2D poses for differential drive robots and planar navigation
- **Data Flow**: Pure transformation - no external dependencies or side effects

### Robotics Applications

- **Differential Drive**: Essential for robots without omnidirectional movement
- **Field Navigation**: Converts 3D camera poses to 2D field coordinates
- **Heading Control**: Preserves directional information for autonomous navigation

## Development Notes

### Extending Functionality

Potential enhancements include:
- **Configurable Axes**: Support for different flattening planes (XZ, YZ)
- **Rotation Mode Selection**: Options for preserving different rotation components
- **Coordinate System Conversion**: Support for different robotics conventions

### Performance Considerations

- **Computational Load**: Minimal trigonometric operations per pose
- **Memory Overhead**: Single matrix copy operation
- **Thread Safety**: Stateless operation suitable for concurrent processing

## Error Handling

### Input Validation

- **Matrix Format**: Assumes proper 4x4 homogeneous transformation matrices
- **Type Safety**: Requires numpy array inputs with appropriate dtypes

### Robustness Features

- **Copy Operations**: Never modifies input data directly
- **Consistent Output**: Always returns valid 4x4 transformation matrix

## Future Enhancements

### Planned Features

- **Multi-Pose Processing**: Batch processing of multiple poses
- **Interpolation Modes**: Different flattening algorithms for various use cases
- **Validation Options**: Configurable input validation strictness
