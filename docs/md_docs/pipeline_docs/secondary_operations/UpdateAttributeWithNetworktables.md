# UpdateAttributeWithNetworktables Operation Overview

## Overview

The `UpdateAttributeWithNetworktables` operation is a secondary pipeline operation that enables dynamic runtime configuration of pipeline operations through NetworkTables. This operation bridges the gap between external robot control systems and the vision pipeline, allowing real-time parameter adjustments without pipeline restarts.

## Architecture

### Dynamic Configuration System

The operation implements a live parameter update mechanism that:

1. **NetworkTables Monitoring**: Continuously reads values from NetworkTables
2. **Operation Discovery**: Locates target operations within the pipeline by class name
3. **Attribute Modification**: Updates operation attributes using reflection-based access
4. **Data Passthrough**: Returns input data unchanged for pipeline chaining

### Naming Convention Handling

The operation includes automatic conversion between snake_case (Python convention) and CamelCase (NetworkTables convention) to ensure seamless integration with FRC robot control systems.

## Key Features

### Dynamic Reconfiguration

- **Runtime Updates**: Modify operation parameters without restarting the pipeline
- **NetworkTables Integration**: Compatible with standard FRC communication protocols
- **Live Parameter Adjustment**: Enable real-time vision system tuning

### Robust Operation Discovery

- **Class Name Resolution**: Locate operations by their class names within the pipeline
- **Attribute Validation**: Verify operations support dynamic attribute modification
- **Error Handling**: Comprehensive error reporting for configuration issues

### Data Flow Transparency

- **Passthrough Operation**: Input data flows through unchanged
- **Non-Blocking Updates**: Configuration changes don't interrupt pipeline execution
- **Exception Safety**: Failed updates don't crash the vision pipeline

## Configuration

### Required Parameters

- **pipeline**: Pipeline instance containing operations to configure
- **network_table**: NetworkTable instance for parameter source
- **action_name**: Name of the operation class to update (snake_case)
- **attribute_name**: Name of the attribute to modify
- **network_table_key**: NetworkTables key to monitor for updates

### Configuration Example

```python
network_updater = UpdateAttributeWithNetworktables(
    pipeline=vision_pipeline,
    network_table=robot_network_table,
    action_name="color_threshold",
    attribute_name="hue_center",
    network_table_key="/vision/color/hue_center"
)
```

## Data Flow

### Processing Flow

1. **NetworkTables Query**: Read current value from specified NetworkTables key
2. **Validity Check**: Ensure retrieved value is not None
3. **Operation Location**: Find target operation by converted class name
4. **Attribute Update**: Modify operation attribute with new value
5. **Data Return**: Return input data unchanged

### Processing Steps

```
Input: Pipeline Data (unchanged)
       ↓
Read value from NetworkTables
       ↓
If value exists:
  Find target operation by name
  Update operation attribute
  Handle any errors gracefully
       ↓
Return input data unchanged
```

## Usage Examples

### Dynamic Color Threshold Adjustment

```python
# Configure for color threshold hue adjustment
hue_updater = UpdateAttributeWithNetworktables(
    pipeline=pipeline,
    network_table=nt_table,
    action_name="color_threshold",
    attribute_name="hue_center",
    network_table_key="/vision/color/hue_center"
)

# In pipeline execution
result = hue_updater.run(frame)
# Operation automatically updates color threshold hue from NetworkTables
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
      "name": "update_attribute_with_networktables",
      "config": {
        "action_name": "color_threshold",
        "attribute_name": "saturation_min",
        "network_table_key": "/vision/color/saturation_min"
      }
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
└── update_attribute_with_networktables.py    # Main operation implementation
```

## Technical Details

### Naming Convention Conversion

**Snake to Camel Case:**
```python
def _snake_to_camel(snake_str: str) -> str:
    components = snake_str.split("_")
    return "".join(word.capitalize() for word in components)

# Example: "color_threshold" -> "ColorThreshold"
```

### Operation Interface Requirements

**Required Methods:**
- `set_attribute(attribute_name, value)`: Method for dynamic attribute modification
- Class name matching for operation discovery

### NetworkTables Integration

- **Data Types**: Currently supports numeric values from NetworkTables
- **Key Monitoring**: Continuous reading from specified NetworkTables keys
- **Update Frequency**: Updates occur on every pipeline execution cycle

## Integration Points

### NetworkTables Integration

- **FRC Compatibility**: Works with standard FRC NetworkTables infrastructure
- **Robot Control Integration**: Enables robot code to adjust vision parameters
- **Real-Time Tuning**: Supports live vision system calibration during matches

### Pipeline Integration

- **Operation Modification**: Provides external control over pipeline behavior
- **Configuration Management**: Enables dynamic parameter adjustment
- **Debugging Support**: Allows runtime parameter experimentation

## Development Notes

### Operation Requirements

- **Attribute Interface**: Target operations must implement `set_attribute` method
- **Class Naming**: Operation class names must be discoverable in pipeline
- **Thread Safety**: Consider thread safety for concurrent NetworkTables access

### Performance Considerations

- **NetworkTables Latency**: Updates depend on NetworkTables communication timing
- **Operation Lookup**: Class name resolution occurs on every execution
- **Error Handling Overhead**: Exception handling for robust operation

## Error Handling

### Configuration Errors

- **Operation Not Found**: Clear error messages when target operation doesn't exist
- **Missing Methods**: Validation that operations support attribute modification
- **NetworkTables Issues**: Graceful handling of NetworkTables communication failures

### Robustness Features

- **Null Value Handling**: Ignores None values from NetworkTables
- **Type Safety**: Validates operation interfaces before modification attempts
- **Exception Isolation**: Failed updates don't interrupt pipeline execution

## Future Enhancements

### Planned Features

- **Multi-Parameter Updates**: Support for updating multiple attributes simultaneously
- **Type Conversion**: Automatic type conversion for different parameter types
- **Configuration Persistence**: Save/restore dynamic configurations
- **Update Scheduling**: Control update frequency and timing
- **Parameter Validation**: Value range and type validation before application
- **Bulk Operations**: Update multiple operations with single NetworkTables entry
