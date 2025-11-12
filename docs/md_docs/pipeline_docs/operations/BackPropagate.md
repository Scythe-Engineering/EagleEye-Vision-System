# Back Propagate Operation

## Overview

The `BackPropagate` is a secondary pipeline operation that enables data flow reversal within the processing pipeline. It takes input data and forwards it to a specified earlier operation in the pipeline via the `back_propagate_input` method, enabling downstream operations to provide feedback or additional context to upstream operations.

## Operation Type

**Secondary Operation** - Pipeline data flow control utility

## Category

`flow` - Data flow control operation

## Input/Output

- **Input**: `Any` - Data to back-propagate to upstream operation
- **Output**: `Any` - Same data passed through unchanged (pass-through behavior)

### Processing Behavior

The operation forwards data to an earlier pipeline operation while maintaining pass-through functionality for the main processing chain. When `target_pipeline_name` is provided, the back-propagated data is dispatched to the specified pipeline instead of the current one.

## Parameters

### Constructor Parameters

- `pipeline` (Pipeline): Reference to the pipeline containing operations
- `action_name` (str): Name of the target operation to receive back-propagated data (in snake_case)
- `target_pipeline_name` (str, optional): Name of another pipeline to receive the back-propagated data; defaults to the current pipeline

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
            "action_name": "back_propagate",
            "action_params": {
                "action_name": "camera_adjust",
                "target_pipeline_name": "visualization_pipeline"
            }
        }
    ]
}
```

## Cross-Pipeline Targeting

Set `target_pipeline_name` when the feedback should be delivered to an operation that resides in another pipeline for the same camera. The value must match the pipeline key defined in `pipeline_config.json`.

### Python Usage Example

```python
from src.secondary_operations.back_propagate import BackPropagate
from src.config.utils.pipeline import Pipeline

# Initialize pipeline with operations
pipeline = Pipeline(...)

# Create back propagation to camera adjustment operation
back_propagate = BackPropagate(
    pipeline=pipeline,
    action_name="camera_adjust"  # Target operation name in snake_case
)

# Example: back-propagate AprilTag detections to camera adjust for visualization
april_tag_detections = april_tag_detector.run(frame)
result = back_propagate.run(april_tag_detections)

# The detections are now available to camera_adjust operation
# while also continuing through the pipeline
```

## Performance Considerations

### Data Flow Architecture

- **Non-blocking**: Back-propagation happens synchronously but doesn't block main processing
- **Reference Passing**: Only passes data references, no expensive copying
- **Thread Safety**: Depends on target operation's thread safety for back_propagate_input

### Pipeline Integration

- **Operation Discovery**: Dynamically resolves target operations by class name
- **Caching**: Caches resolved operations for performance after initial lookup
- **Error Handling**: Validates target operations have required back_propagate_input method

### Memory Management

- **Reference Only**: No additional memory allocation for data copying
- **Cache Management**: Maintains operation references to avoid repeated lookups
- **Lifecycle**: Properly handles pipeline reconfiguration and operation changes

## Use Cases

### Visualization Data Flow

Providing detection results to visualization operations earlier in the pipeline:

```json
{
    "action_name": "back_propagate",
    "action_params": {
        "action_name": "camera_adjust"
    }
}
```

### Temporal Prediction Feedback

Sending pose estimates back to preprocessing operations for prediction improvement:

```json
{
    "action_name": "back_propagate",
    "action_params": {
        "action_name": "temporal_acceleration_preprocessor_rust"
    }
}
```

### Adaptive Parameter Control

Feeding processing results back to adaptive operations for parameter adjustment:

```json
{
    "action_name": "back_propagate",
    "action_params": {
        "action_name": "color_threshold_detection"
    }
}
```

## Implementation Details

### Name Resolution

```python
def _snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to CamelCase for class name matching."""
    components = snake_str.split("_")
    return "".join(word.capitalize() for word in components)
```

The operation converts snake_case action names to CamelCase class names for pipeline resolution.

### Operation Caching

```python
if not self._cache_valid:
    target_operation = self.pipeline.get_operation_by_class_name(self.action_name)
    # Validate and cache operation reference
```

Resolves and caches target operations to avoid repeated pipeline traversals.

### Error Handling

- **Operation Not Found**: Validates target operation exists in pipeline
- **Missing Method**: Ensures target has back_propagate_input method
- **Runtime Errors**: Catches and reports errors during back-propagation calls

## Limitations

1. **Method Dependency**: Target operations must implement back_propagate_input method
2. **Pipeline Structure**: Requires specific pipeline configuration with target operations
3. **Data Type Assumptions**: No validation of data types passed between operations
4. **Synchronous Operation**: Back-propagation happens in the main processing thread
5. **Configuration Changes**: Cache invalidation required when pipeline structure changes

## Visualization

The operation does not provide frame visualization as it is a data flow control operation. The `visualize()` method returns `None`.

### Integration Pattern

Back-propagation is typically used with operations that can benefit from downstream information:

```python
# Detection -> Localization -> Back-propagate to preprocessor
detection_results = detector.run(frame)
pose_estimate = localizer.run(detection_results)
back_propagate.run(pose_estimate)  # Feeds back to preprocessor
```

## Related Operations

- **Camera Adjust**: Receives back-propagated detections for visualization
- **Temporal Acceleration Preprocessor**: Uses back-propagated poses for prediction
- **Any operation with back_propagate_input method**: Can serve as target for back-propagation

## Files

- **Definition**: `src/secondary_operations/back_propagate.py`
