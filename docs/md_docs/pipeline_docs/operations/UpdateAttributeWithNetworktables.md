# Update Attribute With NetworkTables Operation

## Overview

The `UpdateAttributeWithNetworktables` is a secondary pipeline operation that enables dynamic parameter updates through NetworkTables, a communication protocol commonly used in robotics (especially FRC - FIRST Robotics Competition). It allows real-time adjustment of pipeline operation parameters from external sources like robot control systems or monitoring dashboards.

## Operation Type

**Secondary Operation** - Dynamic parameter update utility

## Category

`control` - Parameter control operation

## Input/Output

- **Input**: `Any` - Data to pass through unchanged
- **Output**: `Any` - Same data passed through (pass-through behavior)

### Processing Behavior

Reads values from NetworkTables and updates corresponding operation attributes, while maintaining pass-through data flow.

## Parameters

### Constructor Parameters

- `pipeline` (Pipeline): Pipeline instance containing operations to update
- `network_table` (NetworkTable): NetworkTables instance for communication
- `action_name` (str): Name of the operation to update (in snake_case)
- `attribute_name` (str): Name of the attribute to update on the target operation
- `network_table_key` (str): NetworkTables key to read values from

## Configuration Example

### Pipeline Integration

```json
{
    "object_detection_pipeline": [
        {
            "action_name": "color_threshold_detection",
            "action_params": {
                "target_rgb": [1.0, 0.0, 0.0],
                "threshold": 0.3
            }
        },
        {
            "action_name": "update_attribute_with_networktables",
            "action_params": {
                "action_name": "color_threshold_detection",
                "attribute_name": "threshold",
                "network_table_key": "/vision/color_threshold"
            }
        }
    ]
}
```

### Python Usage Example

```python
from src.secondary_operations.update_attribute_with_networktables import UpdateAttributeWithNetworktables
from src.config.utils.pipeline import Pipeline
from networktables import NetworkTables

# Initialize NetworkTables and pipeline
NetworkTables.initialize(server='roborio-XXXX-frc.local')
vision_table = NetworkTables.getTable('vision')
pipeline = Pipeline(...)

# Create parameter updater
updater = UpdateAttributeWithNetworktables(
    pipeline=pipeline,
    network_table=vision_table,
    action_name="color_threshold_detection",
    attribute_name="threshold",
    network_table_key="color_threshold"
)

# During pipeline execution, threshold will be updated from NetworkTables
# NetworkTables value at "/vision/color_threshold" controls detection sensitivity
result = updater.run(pipeline_input)
```

## Performance Considerations

### Real-time Updates

- **Immediate Effect**: Parameter changes take effect on next pipeline iteration
- **Minimal Overhead**: Only reads from NetworkTables when data is available
- **Type Safety**: Assumes numeric data types from NetworkTables

### Error Handling

- **Graceful Degradation**: Continues operation if NetworkTables read fails
- **Validation**: Checks for operation existence and attribute availability
- **Logging**: Provides clear error messages for debugging

### Network Efficiency

- **On-demand Reading**: Only reads when operation is executed
- **No Polling**: Uses NetworkTables' built-in update mechanisms
- **Lightweight**: Minimal network and processing overhead

## Use Cases

### Dynamic Color Detection Tuning

Adjusting color threshold parameters during robot operation:

```json
{
    "action_name": "update_attribute_with_networktables",
    "action_params": {
        "action_name": "color_threshold_detection",
        "attribute_name": "threshold",
        "network_table_key": "/vision/color_threshold"
    }
}
```

### Camera Parameter Adjustment

Real-time camera control through robot control systems:

```json
{
    "action_name": "update_attribute_with_networktables",
    "action_params": {
        "action_name": "camera_adjust",
        "attribute_name": "exposure",
        "network_table_key": "/vision/camera_exposure"
    }
}
```

### Localization Parameter Tuning

Adjusting pose estimation parameters for different environments:

```json
{
    "action_name": "update_attribute_with_networktables",
    "action_params": {
        "action_name": "pnp_camera_localization",
        "attribute_name": "jump_threshold",
        "network_table_key": "/vision/pose_jump_threshold"
    }
}
```

## Implementation Details

### Name Resolution

```python
def _snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to CamelCase for operation lookup."""
    components = snake_str.split("_")
    return "".join(word.capitalize() for word in components)
```

### Parameter Update Logic

```python
def run(self, passthrough_data: Any) -> Any:
    data = self.network_table.getNumber(self.network_table_key, None)
    if data is not None:
        action_object = self.pipeline.get_operation_by_class_name(self.action_name)
        action_object.set_attribute(self.attribute_name, data)
    return passthrough_data
```

### Error Handling

- **NetworkTables Read**: Gracefully handles unavailable keys
- **Operation Lookup**: Validates operation exists in pipeline
- **Attribute Access**: Ensures target operation has set_attribute method

## NetworkTables Integration

### FRC Robotics Context

NetworkTables is the standard communication protocol for FRC robots, allowing:

- **Driver Station Communication**: Real-time parameter adjustment from driver interfaces
- **Autonomous Tuning**: Dynamic parameter updates during autonomous routines
- **Teleoperation Support**: Live adjustments during human-controlled operation

### Data Types

Currently supports numeric values from NetworkTables:
- **Integer**: For discrete parameter values
- **Float**: For continuous parameter ranges
- **Default Handling**: Uses None as default when key unavailable

## Limitations

1. **Numeric Only**: Currently only supports numeric parameter updates
2. **NetworkTables Dependency**: Requires NetworkTables infrastructure
3. **Operation Compatibility**: Target operations must implement set_attribute method
4. **No Validation**: Assumes parameter values are within valid ranges
5. **Single Value**: Updates one parameter per operation instance

## Visualization

The operation does not provide frame visualization as it is a parameter control operation. The `visualize()` method returns `None`.

### Monitoring Integration

Consider integrating with monitoring to track:
- Parameter update frequency
- Value ranges and changes
- Update success/failure rates
- NetworkTables connection status

## Related Operations

- **Operations with set_attribute**: Any operation supporting dynamic parameter updates
- **NetworkTables Consumers**: Other operations reading from NetworkTables
- **Parameter Control**: Alternative parameter adjustment mechanisms

## Files

- **Definition**: `src/secondary_operations/update_attribute_with_networktables.py`
