# Tag Filter Operation

## Overview

The `TagFilter` is a secondary pipeline operation that filters AprilTag detections based on configurable whitelist or blacklist criteria. It allows selective processing of specific AprilTags while excluding others, enabling targeted localization and tracking in environments with multiple tags.

## Operation Type

**Secondary Operation** - Detection filtering utility

## Category

`filter` - Data filtering operation

## Input/Output

- **Input**: `List[Detection]` or `List[CustomDetection]` or `None` - AprilTag detections
- **Output**: `List[Detection]` or `List[CustomDetection]` or `None` - Filtered detections

### Processing Behavior

Filters detections based on tag ID using either whitelist (inclusive) or blacklist (exclusive) modes.

## Parameters

### Constructor Parameters

- `filter_mode` (str): "whitelist" or "blacklist" filtering mode (default: "whitelist")
- `tag_ids` (List[int]): List of tag IDs to filter by (default: None, becomes empty list)

### Filter Modes

- **Whitelist Mode**: Only tags with IDs in the list are kept (empty list keeps all tags)
- **Blacklist Mode**: Tags with IDs in the list are removed

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
            "action_name": "tag_filter",
            "action_params": {
                "filter_mode": "whitelist",
                "tag_ids": [0, 1, 2, 5, 10]
            }
        }
    ]
}
```

### Python Usage Example

```python
from src.secondary_operations.tag_filter import TagFilter

# Initialize whitelist filter for specific navigation tags
tag_filter = TagFilter(
    filter_mode="whitelist",
    tag_ids=[0, 1, 2, 5, 10]  # Only keep these tag IDs
)

# Example detections from AprilTag detector
detections = [
    # Detection objects with tag_id, corners, etc.
]

# Filter detections
filtered_detections = tag_filter.run(detections)

print(f"Original detections: {len(detections)}")
print(f"Filtered detections: {len(filtered_detections) if filtered_detections else 0}")
```

## Performance Considerations

### Efficient Filtering

- **Set-based Lookup**: O(1) tag ID checking using Python sets
- **Minimal Overhead**: Linear pass through detections with constant-time filtering
- **Memory Efficient**: Only stores tag ID set, not full detection objects

### Thread Safety

- **Input Caching**: Thread-safe storage of input detections for visualization
- **Lock Protection**: Proper locking for concurrent access to cached data

### Scalability

- **Linear Performance**: Processing time scales linearly with number of detections
- **Memory Bounded**: Memory usage depends only on configured tag ID list size

## Tuning Guide

### Filter Mode Selection

#### Whitelist Mode (Recommended for most applications)
- **Precise Control**: Only processes known, trusted tags
- **Security**: Prevents processing of unexpected or uncalibrated tags
- **Performance**: Reduces downstream processing load
- **Use Case**: Production robots with known tag deployments

#### Blacklist Mode
- **Flexibility**: Allows processing of any tags except known problematic ones
- **Exploration**: Useful during mapping and calibration phases
- **Maintenance**: Easy to exclude specific malfunctioning tags
- **Use Case**: Development and testing environments

### Tag ID Configuration

1. **Mapping-Based**: Use tag IDs that correspond to physical locations
2. **Priority-Based**: Include high-priority navigation tags first
3. **Testing**: Start with small lists and expand based on requirements
4. **Documentation**: Maintain mapping between tag IDs and physical locations

## Use Cases

### Selective Robot Navigation

Focus on specific navigation tags while ignoring others:

```json
{
    "filter_mode": "whitelist",
    "tag_ids": [0, 1, 2, 10, 11, 12]
}
```

### Quality Control

Exclude known damaged or poorly positioned tags:

```json
{
    "filter_mode": "blacklist",
    "tag_ids": [7, 15, 23]  // Damaged tags to exclude
}
```

### Multi-Robot Coordination

Different robots can focus on different tag subsets:

```json
{
    "filter_mode": "whitelist",
    "tag_ids": [100, 101, 102, 103, 104]  // Robot A's tags
}
```

## Implementation Details

### Filtering Logic

```python
if self.filter_mode == "whitelist":
    # Keep tags in whitelist, or all if empty
    if not self.tag_ids or tag_id in self.tag_ids:
        filtered_detections.append(detection)
elif self.filter_mode == "blacklist":
    # Remove tags in blacklist
    if tag_id not in self.tag_ids:
        filtered_detections.append(detection)
```

### Configuration Updates

```python
def update_config(self, json_config: dict) -> None:
    if "filter_mode" in json_config:
        # Validate and update mode
    if "tag_ids" in json_config:
        self.tag_ids = set(json_config["tag_ids"])  # Convert to set
```

## Visualization

The operation provides comprehensive visualization of filtering decisions:

### Features

- **Color Coding**: Green boxes for kept tags, red boxes for excluded tags
- **Tag ID Labels**: White text showing tag identification numbers
- **Bounding Boxes**: Corner-based polygons around detected tags
- **Real-time**: Visualizes the most recent detection results

### Usage

```python
tag_filter = TagFilter(...)

# Run filtering
filtered = tag_filter.run(detections)

# Visualize filtering decisions
visualized_frame = tag_filter.visualize(frame)
```

### Legend

- **Green Boxes**: Tags that pass through the filter
- **Red Boxes**: Tags that are excluded by the filter
- **White Text**: Tag ID numbers for identification

## Limitations

1. **ID-Based Only**: Filtering only considers tag IDs, not position or quality
2. **Static Configuration**: Requires reconfiguration to change filter criteria
3. **No Dynamic Rules**: Cannot filter based on runtime conditions
4. **Memory Storage**: Caches input detections for visualization memory usage

## Related Operations

- `DetectApriltagsDefinition`: Provides detections for filtering
- `PnpCameraLocalizationDefinition`: Uses filtered detections for pose estimation
- `CameraAdjust`: Can visualize filtered detections when back-propagated

## Files

- **Definition**: `src/secondary_operations/tag_filter.py`
