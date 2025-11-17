# FpsLimiter Operation Overview

## Overview

The `FpsLimiter` operation is a secondary pipeline operation that controls processing frame rate by introducing controlled delays between pipeline executions. This operation ensures consistent timing and prevents resource exhaustion in vision processing pipelines while maintaining real-time performance requirements.

## Architecture

### Timing Control Mechanism

The operation implements frame rate limiting through:

1. **Time Tracking**: Records timestamps of each pipeline execution
2. **Interval Calculation**: Computes required sleep time to maintain target FPS
3. **Controlled Sleeping**: Uses system sleep functions to delay processing
4. **Frame Passthrough**: Returns input frame unchanged after timing adjustment

### Integration Design

The operation integrates with the EagleEye web interface for potential future visualization and monitoring capabilities, while maintaining minimal overhead for the core timing functionality.

## Key Features

### Precise Rate Control

- **Target FPS Enforcement**: Maintains exact frame rates through timing calculations
- **Adaptive Sleeping**: Only sleeps when necessary to achieve target rate
- **No Frame Dropping**: Processes every frame but controls execution timing

### Minimal Overhead

- **Frame Passthrough**: Input frames are returned unmodified
- **Lightweight Operation**: Minimal computational overhead beyond timing
- **Memory Efficient**: No additional data structures or frame copies

### Real-Time Performance

- **Consistent Timing**: Provides stable, predictable frame intervals
- **System Integration**: Uses standard system timing functions
- **Resource Management**: Prevents excessive CPU usage in vision pipelines

## Configuration

### Required Parameters

- **fps**: Target frames per second (float)
- **web_interface**: EagleEyeInterface instance for integration

### Configuration Example

```python
fps_limiter = FpsLimiter(
    web_interface=eagle_eye_interface,
    fps=30.0  # 30 FPS target
)
```

## Data Flow

### Processing Flow

1. **Time Recording**: Capture current system time
2. **Elapsed Calculation**: Compute time since last execution
3. **Sleep Determination**: Calculate required sleep to maintain target FPS
4. **Conditional Sleeping**: Sleep only if needed to meet timing requirements
5. **Frame Return**: Return input frame unchanged

### Timing Logic

```
Current Time ──→ Calculate Elapsed ──→ Target Interval
      ↑                                        ↓
      └───────────── Sleep if needed ──────────┘
                           ↓
                    Update Last Time
                           ↓
                     Return Frame
```

## Usage Examples

### Basic FPS Limiting

```python
# Initialize with 15 FPS target
fps_limiter = FpsLimiter(web_interface, fps=15.0)

# In pipeline loop
while processing:
    frame = camera.read()
    processed_frame = fps_limiter.run(frame)
    # Pipeline continues at exactly 15 FPS
```

### Pipeline Integration

```json
{
  "operations": [
    {
      "type": "primary",
      "name": "camera_input"
    },
    {
      "type": "secondary",
      "name": "fps_limiter",
      "config": {
        "fps": 20.0
      }
    },
    {
      "type": "secondary",
      "name": "object_detection"
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
└── fps_limiter.py    # Main operation implementation
```

## Technical Details

### Timing Implementation

**Interval Calculation:**
```
target_interval = 1.0 / target_fps
elapsed_time = current_time - last_run_time
sleep_time = target_interval - elapsed_time
```

**Sleep Execution:**
```python
if sleep_time > 0:
    time.sleep(sleep_time)
```

### Precision Considerations

- **System Timer Resolution**: Depends on underlying OS timing precision
- **Sleep Accuracy**: System sleep functions may have minimum granularity
- **Drift Compensation**: Automatic adjustment for timing variations

## Integration Points

### Pipeline Integration

- **Position Flexibility**: Can be placed anywhere in the pipeline sequence
- **Frame Processing**: Operates on video frames but doesn't modify content
- **Resource Management**: Helps prevent pipeline resource exhaustion

### Web Interface Integration

- **Monitoring Potential**: Web interface reference for future visualization
- **Status Reporting**: Could report timing statistics and performance metrics
- **Configuration Updates**: Potential for runtime FPS adjustments

## Development Notes

### Performance Characteristics

- **CPU Usage**: Minimal when meeting timing requirements
- **Memory Footprint**: Negligible additional memory usage
- **Thread Behavior**: Blocking operation during sleep periods

### Extension Possibilities

- **Dynamic FPS**: Runtime adjustable frame rates
- **Performance Monitoring**: Detailed timing statistics collection
- **Quality Adaptation**: Frame rate adjustment based on processing load

## Error Handling

### Timing Edge Cases

- **First Run**: No sleep on initial execution (no previous timing data)
- **Negative Sleep**: Handles cases where processing exceeds target interval
- **System Time Changes**: Robust against system clock adjustments

### Robustness Features

- **Exception Safety**: Timing failures don't crash the pipeline
- **Graceful Degradation**: Continues operation even with timing precision issues

## Future Enhancements

### Planned Features

- **Adaptive Rate Control**: Dynamic FPS based on system load
- **Performance Metrics**: Detailed timing and frame rate statistics
- **Quality-Based Limiting**: Adjust rates based on detection quality requirements
- **Multi-Rate Support**: Different FPS limits for different pipeline branches
