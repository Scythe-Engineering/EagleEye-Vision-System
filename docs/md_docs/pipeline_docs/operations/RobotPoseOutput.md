# Robot Pose Output Operation

## Overview

The `RobotPoseOutput` is a secondary pipeline operation that publishes robot pose estimates to the EagleEye web interface. It efficiently manages pose updates by avoiding redundant transmissions of identical pose data, enabling real-time robot position visualization and monitoring.

## Operation Type

**Secondary Operation** - Data output and communication utility

## Category

`output` - Data output operation

## Input/Output

- **Input**: `np.ndarray` - 4x4 homogeneous transformation matrix representing robot pose
- **Output**: `None` - No return value (output-only operation)

### Processing Behavior

Receives pose data and publishes it to the web interface, with built-in deduplication to prevent unnecessary network traffic.

## Parameters

### Constructor Parameters

- `web_interface` (EagleEyeInterface): Web interface instance for publishing pose updates

## Configuration Example

### Pipeline Integration

```json
{
    "object_detection_pipeline": [
        {
            "action_name": "detect_apriltags",
            "action_params": {}
        },
        {
            "action_name": "pnp_camera_localization",
            "action_params": {
                "camera_parameters_path": "config/camera.json",
                "apriltag_map_path": "config/tags.fmap"
            }
        },
        {
            "action_name": "robot_pose_output",
            "action_params": {}
        }
    ]
}
```

### Python Usage Example

```python
from src.secondary_operations.robot_pose_output import RobotPoseOutput
from src.webui.web_server import EagleEyeInterface
import numpy as np

# Initialize web interface and pose output
web_interface = EagleEyeInterface()
pose_output = RobotPoseOutput(web_interface=web_interface)

# Example robot pose (4x4 transformation matrix)
robot_pose = np.eye(4)
robot_pose[0, 3] = 2.5   # X position in meters
robot_pose[1, 3] = 1.8   # Y position in meters
robot_pose[2, 3] = 0.0   # Z position in meters

# Publish pose to web interface
pose_output.run(robot_pose)

# Identical poses are automatically deduplicated
pose_output.run(robot_pose)  # No second update sent
```

## Performance Considerations

### Network Efficiency

- **Deduplication**: Prevents redundant pose transmissions using numpy array comparison
- **Memory Management**: Stores only the last sent pose for comparison
- **Real-time Updates**: Immediate transmission of new pose data

### Integration Design

- **Web Interface Coupling**: Direct integration with EagleEye web server
- **Thread Safety**: Assumes web interface handles concurrent access appropriately
- **Error Resilience**: Failures in pose transmission don't affect pipeline flow

### Resource Usage

- **Minimal Overhead**: Only array comparison and web interface calls
- **Memory Bounded**: Constant memory usage regardless of update frequency
- **Network Conscious**: Reduces unnecessary network traffic

## Implementation Details

### Pose Deduplication

```python
def run(self, pose: np.ndarray) -> None:
    if self._last_sent_pose is not None and np.array_equal(
        self._last_sent_pose, pose
    ):
        return None  # Skip duplicate transmission

    self.web_interface.update_robot_position(pose)
    self._last_sent_pose = pose.copy()
```

### State Management

- **Last Pose Tracking**: Maintains copy of most recently transmitted pose
- **Exact Comparison**: Uses numpy array equality for precise deduplication
- **Update on Change**: Only transmits when pose actually differs

## Use Cases

### Real-time Robot Monitoring

Publishing pose updates for live robot position tracking:

```json
{
    "action_name": "robot_pose_output",
    "action_params": {}
}
```

### Teleoperation Support

Providing pose feedback for remote robot control:

```json
{
    "action_name": "robot_pose_output",
    "action_params": {}
}
```

### Multi-Robot Coordination

Sharing pose information across multiple robot systems:

```json
{
    "action_name": "robot_pose_output",
    "action_params": {}
}
```

## Integration with Web Interface

### Update Mechanism

The operation calls `web_interface.update_robot_position(pose)` to publish pose data. The web interface is responsible for:

- Broadcasting pose updates to connected clients
- Maintaining pose history for visualization
- Handling network communication and client management

### Data Format

Poses are transmitted as 4x4 homogeneous transformation matrices in the global coordinate system, providing complete 6DOF pose information (position and orientation).

## Limitations

1. **Web Interface Dependency**: Requires running EagleEye web server instance
2. **Network Reliability**: Pose updates depend on network connectivity
3. **Memory Accumulation**: Stores last pose indefinitely (minimal impact)
4. **No Error Handling**: Failures in web interface calls are not handled locally
5. **No Buffering**: Lost updates are not retransmitted

## Visualization

The operation does not provide frame visualization as it is an output-only operation. The `visualize()` method returns `None`.

### Web Interface Visualization

Pose data published by this operation enables:

- **Real-time Position Display**: Live robot position on maps/floor plans
- **Pose History Trails**: Visualization of robot movement paths
- **Coordinate System Display**: Showing robot orientation and heading
- **Multi-client Updates**: Broadcasting to multiple connected interfaces

## Related Operations

- **Localization Operations**: `PnpCameraLocalizationDefinition`, `FusedCameraLocalizationDefinition`
- **Filtering Operations**: `PoseOutlierFilterRust` for preprocessing poses
- **Web Interface**: Consumes pose data for visualization and monitoring

## Files

- **Definition**: `src/secondary_operations/robot_pose_output.py`
- **Web Interface**: `src/webui/web_server.py`
