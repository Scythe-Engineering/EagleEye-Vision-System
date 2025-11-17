# TagFilter Operation Overview

## Overview

The `TagFilter` operation is a secondary pipeline operation that provides selective filtering of AprilTag detections based on tag ID whitelisting or blacklisting. This operation enables robots to focus on specific tags of interest while ignoring irrelevant ones, improving processing efficiency and reducing false positives in vision-based navigation systems.

## Architecture

### Selective Filtering System

The operation implements flexible tag selection through:

1. **Mode Selection**: Choose between whitelist (inclusive) or blacklist (exclusive) filtering
2. **ID Matching**: Fast set-based lookup for tag ID filtering decisions
3. **Data Preservation**: Maintains all detection data for filtered tags
4. **Visualization Support**: Thread-safe storage of input detections for visual feedback

### Thread-Safe Design

The operation includes thread-safe detection storage to support concurrent visualization and processing in multi-threaded pipeline environments.

## Key Features

### Flexible Filtering Modes

- **Whitelist Mode**: Only allow tags with specified IDs to pass through
- **Blacklist Mode**: Exclude tags with specified IDs while allowing others
- **Empty List Handling**: Whitelist mode with empty list allows all tags
- **Dynamic Configuration**: Runtime adjustment of filter parameters

### High-Performance Operation

- **Set-Based Lookup**: O(1) average-case tag ID checking
- **Minimal Overhead**: Lightweight filtering with no data transformation
- **Memory Efficient**: Shared detection storage for input and visualization
- **Type Agnostic**: Supports both Detection and CustomDetection objects

### Rich Visualization

- **Color-Coded Display**: Green boxes for kept tags, red for excluded tags
- **ID Annotation**: Tag IDs displayed at detection centers
- **Real-Time Feedback**: Live visualization of filtering decisions
- **Thread-Safe Access**: Protected visualization data for multi-threaded use

## Configuration

### Required Parameters

- **filter_mode**: Filtering mode - "whitelist" or "blacklist"
- **tag_ids**: List of tag IDs to filter by (empty list for whitelist = allow all)

### Configuration Example

```python
# Allow only specific navigation tags
nav_filter = TagFilter(
    filter_mode="whitelist",
    tag_ids=[1, 2, 3, 4]  # FRC field perimeter tags
)

# Block known problem tags
problem_filter = TagFilter(
    filter_mode="blacklist",
    tag_ids=[99, 100]  # Known faulty tag IDs
)
```

## Data Flow

### Processing Flow

1. **Detection Input**: Receive list of AprilTag detections
2. **Storage**: Thread-safely store detections for visualization
3. **Filtering**: Apply whitelist/blacklist logic to each detection
4. **Output**: Return filtered detection list maintaining original data

### Processing Steps

```
Input: List[Detection/CustomDetection] or None
       ↓
Store detections for visualization (thread-safe)
       ↓
If input is None: return None
       ↓
For each detection:
  Check tag_id against filter criteria
  Include/exclude based on filter_mode
       ↓
Output: Filtered detection list
```

## Usage Examples

### Strategic Tag Selection

```python
# Filter for only field perimeter tags
field_tags = TagFilter(
    filter_mode="whitelist",
    tag_ids=[1, 2, 3, 4, 5, 6]  # Standard FRC field tags
)

# Process detections
raw_detections = apriltag_detector.run(frame)
filtered_detections = field_tags.run(raw_detections)
# Result: Only perimeter tags remain for pose estimation
```

### Problem Tag Exclusion

```python
# Remove unreliable tags
reliable_tags = TagFilter(
    filter_mode="blacklist",
    tag_ids=[15, 23, 47]  # Known problematic tags
)

clean_detections = reliable_tags.run(all_detections)
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
      "name": "tag_filter",
      "config": {
        "filter_mode": "whitelist",
        "tag_ids": [1, 2, 3, 8, 9, 10]
      }
    },
    {
      "type": "secondary",
      "name": "pose_estimation"
    }
  ]
}
```

## Directory Structure

```
src/secondary_operations/
└── tag_filter.py    # Main operation implementation
```

## Technical Details

### Filtering Logic

**Whitelist Mode:**
```python
if not self.tag_ids or tag_id in self.tag_ids:
    keep_detection = True
```

**Blacklist Mode:**
```python
if tag_id not in self.tag_ids:
    keep_detection = True
```

### Thread Safety Implementation

- **Lock Protection**: Mutex around detection storage for visualization
- **Read-Copy Pattern**: Visualization reads snapshot of stored detections
- **Minimal Contention**: Lock held only during storage/retrieval operations

## Integration Points

### Pipeline Integration

- **Detection Refinement**: Filters detections before pose estimation operations
- **Performance Optimization**: Reduces processing load by eliminating irrelevant tags
- **Strategy Implementation**: Enables robots to focus on mission-critical tags

### FRC Applications

- **Field Navigation**: Filter for perimeter tags during navigation
- **Target Acquisition**: Allow only game piece or goal tags
- **Multi-Robot Coordination**: Different robots can focus on different tag sets

## Development Notes

### Tag ID Management

- **FRC Standards**: Compatible with official FRC AprilTag field layouts
- **Dynamic Updates**: Runtime filter reconfiguration for strategy changes
- **Validation**: Input validation ensures proper filter mode and ID list format

### Performance Considerations

- **Lookup Efficiency**: Set-based ID checking scales well with large ID lists
- **Memory Overhead**: Minimal additional memory for set storage
- **Visualization Cost**: Drawing operations performed only when visualizing

## Error Handling

### Configuration Validation

- **Mode Validation**: Ensures filter_mode is "whitelist" or "blacklist"
- **ID List Handling**: Accepts None as empty list, converts to set for efficiency
- **Type Safety**: Handles various detection object types gracefully

### Robustness Features

- **None Handling**: Proper handling of None input detections
- **Exception Safety**: Filtering failures don't crash the pipeline
- **Data Integrity**: Preserves all detection data for filtered results

## Future Enhancements

### Planned Features

- **Pattern-Based Filtering**: Support for tag ID ranges and patterns
- **Priority-Based Selection**: Tag importance weighting for selective processing
- **Dynamic Filtering**: Rules-based filtering with environmental context
- **Filter Chains**: Multiple sequential filters with different criteria
- **Performance Metrics**: Filtering statistics and efficiency monitoring
- **Tag Health Monitoring**: Automatic detection of problematic tags for blacklisting
