# FPS Limiter Operation

## Overview

The `FpsLimiter` is a secondary pipeline operation that controls frame processing rate by introducing controlled delays. It maintains a target frames-per-second (FPS) rate by measuring time between operations and sleeping for the remaining interval needed to achieve the desired frame rate.

## Operation Type

**Secondary Operation** - Pipeline utility operation

## Category

`util` - Utility operation

## Input/Output

- **Input**: `np.ndarray` (BGR image frame)
- **Output**: `np.ndarray` (same frame, unchanged)

### Processing Behavior

The operation passes the input frame through unchanged while controlling processing timing to maintain the target FPS.

## Parameters

### Constructor Parameters

- `web_interface` (EagleEyeInterface): Web interface instance (currently unused)
- `fps` (float): Target frames per second to maintain

## Configuration Example

### Pipeline Integration

The FPS limiter is typically added as a secondary operation in pipeline configuration:

```json
{
    "object_detection_pipeline": [
        {
            "action_name": "detect_apriltags",
            "action_params": {}
        },
        {
            "action_name": "fps_limiter",
            "action_params": {
                "fps": 30.0
            }
        }
    ]
}
```

### Python Usage Example

```python
from src.secondary_operations.fps_limiter import FpsLimiter
from src.webui.web_server import EagleEyeInterface
import cv2

web_interface = EagleEyeInterface()
fps_limiter = FpsLimiter(web_interface=web_interface, fps=30.0)

frame = cv2.imread("input.jpg")

# Process frame with FPS limiting
processed_frame = fps_limiter.run(frame)
# Frame is identical but timing is controlled
```

## Performance Considerations

### Timing Control

- **Precision**: Uses high-resolution system timers for accurate delay calculation
- **Overhead**: Minimal computational overhead beyond timing measurements
- **Accuracy**: Achieves target FPS within system timer resolution limits

### Use Cases

- **Resource Management**: Prevents excessive CPU/GPU usage in continuous processing
- **Network Bandwidth**: Controls data transmission rates in streaming applications
- **Battery Life**: Reduces power consumption on mobile/embedded platforms
- **Thermal Management**: Prevents overheating during sustained operation

## Tuning Guide

### FPS Selection

1. **Real-time Requirements**: Set to match display refresh rates (30, 60 FPS)
2. **Processing Capacity**: Choose based on available computational resources
3. **Application Needs**: Balance between responsiveness and resource usage

### Integration Points

- **Pipeline Start**: Limit input frame rate at the beginning of processing
- **Pipeline End**: Control output rate before display or transmission
- **Multiple Limiters**: Can use different FPS limits at different pipeline stages

## Implementation Details

### Timing Mechanism

```python
def run(self, frame: np.ndarray) -> np.ndarray:
    current_time = time.time()

    if self.last_run_time is not None:
        elapsed_time = current_time - self.last_run_time
        sleep_time = self.target_interval_seconds - elapsed_time

        if sleep_time > 0:
            time.sleep(sleep_time)

    self.last_run_time = time.time()
    return frame
```

### State Management

- Tracks the timestamp of the last processed frame
- Calculates required sleep time to maintain target interval
- Updates timing reference after each operation

## Limitations

1. **System Timer Resolution**: Limited by OS timer precision (typically milliseconds)
2. **Processing Variation**: Cannot compensate for variable processing times exceeding target interval
3. **External Delays**: Does not account for delays introduced by other pipeline operations
4. **No Frame Dropping**: Buffers frames rather than dropping them when running behind

## Visualization

The FPS limiter does not provide frame visualization. The `visualize()` method returns the input frame unchanged.

### Monitoring Integration

Consider integrating with monitoring systems to track:
- Actual vs target FPS achievement
- Sleep time statistics
- Processing time variance

## Related Operations

- **Primary Operations**: Can be combined with any frame-processing operation
- **Other Utilities**: Complements other utility operations for pipeline management

## Files

- **Definition**: `src/secondary_operations/fps_limiter.py`
- **Pipeline Config Example**: `src/config/pipeline_config.json`
