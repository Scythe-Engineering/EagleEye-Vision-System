# BackPropagate Operation Overview

## Overview

The `BackPropagate` operation is a secondary pipeline operation that enables data flow reversal within vision pipelines. This operation allows processed data to be sent backwards to earlier pipeline operations, enabling dynamic feedback loops and cross-pipeline communication in complex vision processing systems.

## Architecture

### Reverse Data Flow System

The operation implements a sophisticated back-propagation mechanism that:

1. **Operation Discovery**: Locates target operations within the same or different pipelines
2. **Method Invocation**: Calls specialized `back_propagate_input` methods on target operations
3. **Data Routing**: Enables data to flow backwards through pipeline processing stages
4. **Caching Optimization**: Maintains cached references to target operations for performance

### Cross-Pipeline Communication

The operation supports inter-pipeline communication, allowing data processed in one pipeline to influence operations in another pipeline, enabling coordinated multi-camera vision systems.

## Key Features

### Dynamic Feedback Loops

- **Reverse Data Flow**: Send processed data backwards to earlier operations
- **Pipeline Coordination**: Enable communication between different pipelines
- **Real-Time Adaptation**: Support dynamic pipeline reconfiguration based on results

### Robust Operation Discovery

- **Class Name Resolution**: Locate operations by class name across pipelines
- **Interface Validation**: Verify target operations support back-propagation
- **Caching System**: Optimize operation lookups for repeated executions

### Error Handling

- **Graceful Failures**: Comprehensive error reporting for configuration issues
- **Cache Management**: Automatic cache invalidation on configuration changes
- **Exception Safety**: Failed back-propagation doesn't interrupt main pipeline flow

## Configuration

### Required Parameters

- **pipeline**: Source pipeline containing this operation
- **action_name**: Name of target operation class (snake_case)

### Optional Parameters

- **target_pipeline_name**: Name of target pipeline (defaults to current pipeline)

### Configuration Example

```python
back_prop = BackPropagate(
    pipeline=current_pipeline,
    action_name="color_threshold",
    target_pipeline_name="secondary_pipeline"
)
```

## Data Flow

### Processing Flow

1. **Cache Validation**: Check if target operation reference is still valid
2. **Pipeline Resolution**: Locate target pipeline (current or specified)
3. **Operation Discovery**: Find target operation by class name
4. **Interface Verification**: Confirm operation supports back-propagation
5. **Data Transmission**: Call back_propagate_input on target operation
6. **Data Passthrough**: Return input data unchanged

### Processing Steps

```
Input: Processing Data
       ↓
Resolve target pipeline
       ↓
Find target operation by name
       ↓
Validate back_propagate_input method
       ↓
Call back_propagate_input(data)
       ↓
Return input data unchanged
```

## Usage Examples

### Feedback Loop Creation

```python
# Create back-propagation to color threshold operation
feedback_loop = BackPropagate(
    pipeline=vision_pipeline,
    action_name="color_threshold"
)

# In pipeline execution
processed_data = feedback_loop.run(current_detections)
# Color threshold operation receives detection feedback
```

### Cross-Pipeline Communication

```python
# Send data from main pipeline to secondary pipeline
cross_pipeline_comm = BackPropagate(
    pipeline=main_pipeline,
    action_name="pose_estimator",
    target_pipeline_name="tracking_pipeline"
)

# Data flows from main to tracking pipeline
result = cross_pipeline_comm.run(pose_data)
```

### Pipeline Integration

```json
{
  "operations": [
    {
      "type": "primary",
      "name": "object_detection"
    },
    {
      "type": "secondary",
      "name": "back_propagate",
      "config": {
        "action_name": "apriltag_detector",
        "target_pipeline_name": "pose_pipeline"
      }
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
└── back_propagate.py    # Main operation implementation
```

## Technical Details

### Naming Convention Handling

**Snake to Camel Case Conversion:**
```python
def _snake_to_camel(snake_str: str) -> str:
    components = snake_str.split("_")
    return "".join(word.capitalize() for word in components)

# Example: "color_threshold" -> "ColorThreshold"
```

### Operation Interface Requirements

**Required Method:**
- `back_propagate_input(data)`: Method to receive back-propagated data

### Caching Mechanism

- **Performance Optimization**: Avoid repeated operation lookups
- **Cache Invalidation**: Automatic reset on configuration changes
- **Thread Safety**: Consider thread safety for multi-pipeline environments

## Integration Points

### Pipeline Integration

- **Feedback Loops**: Enable operations to adapt based on downstream results
- **Multi-Pipeline Systems**: Support coordinated processing across camera pipelines
- **Dynamic Behavior**: Allow runtime pipeline reconfiguration

### Operation Interface

- **Back-Propagation Support**: Operations can implement `back_propagate_input` method
- **Data Format Flexibility**: Support arbitrary data formats for back-propagation
- **Exception Handling**: Robust error handling in back-propagation methods

## Development Notes

### Operation Requirements

- **Interface Implementation**: Target operations must implement `back_propagate_input`
- **Data Format Compatibility**: Ensure data formats are compatible between operations
- **Thread Safety**: Consider synchronization for multi-threaded pipeline execution

### Performance Considerations

- **Lookup Overhead**: Initial operation discovery may have latency
- **Caching Benefits**: Subsequent executions benefit from cached references
- **Memory Usage**: Minimal additional memory for operation references

## Error Handling

### Configuration Errors

- **Pipeline Not Found**: Clear errors when target pipelines don't exist
- **Operation Missing**: Validation when target operations are not found
- **Interface Mismatch**: Detection of operations without back-propagation support

### Runtime Errors

- **Method Call Failures**: Comprehensive error reporting for back-propagation failures
- **Exception Isolation**: Failed back-propagation doesn't crash the pipeline
- **Cache Recovery**: Automatic cache rebuilding after errors

## Future Enhancements

### Planned Features

- **Bulk Back-Propagation**: Send data to multiple target operations simultaneously
- **Conditional Propagation**: Support for conditional back-propagation based on data content
- **Propagation Chains**: Multi-hop back-propagation through operation networks
- **Data Transformation**: Support for data format conversion during back-propagation
- **Performance Monitoring**: Metrics and monitoring for back-propagation efficiency
- **Configuration Templates**: Predefined back-propagation patterns for common use cases
