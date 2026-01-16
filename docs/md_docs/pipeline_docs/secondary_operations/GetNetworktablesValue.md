## GetNetworktablesValue

Read data from NetworkTable and output it to downstream operations.

### Description

`GetNetworktablesValue` is a data source operation that reads a numeric value from NetworkTables and outputs it to downstream pipeline operations. Unlike regular operations, data source operations have no input connections and generate their own data independently.

### Usage

This operation is useful for integrating external sensor data from NetworkTables into your vision pipeline. For example, you can read robot state values (like arm position, shooter angle, etc.) and use them to adjust vision processing parameters.

### Configuration

| Parameter           | Type | Description                        | Default | Required |
| ------------------- | ---- | ---------------------------------- | ------- | -------- |
| `network_table_key` | str  | Key to read from the network table | `""`    | Yes      |

### Example Pipeline Configuration

```json
{
    "action_name": "get_networktables_value.py",
    "action_params": {
        "network_table_key": "/SmartDashboard/ShooterAngle"
    },
    "position": {
        "x": 300,
        "y": 200
    }
}
```

### Data Source Behavior

As a data source operation:

- No input connections are required
- The `run()` method receives `None` as input (ignored)
- The operation executes one timestep before its data is needed to get the most up-to-date value possible
- Output is available to downstream operations at the next timestep

### Output

The operation returns the numeric value read from NetworkTables, or `None` if the key is not found.

### Integration Example

```json
{
    "CAM0": [
        {
            "action_name": "device_input.py",
            "action_params": {},
            "position": { "x": 100, "y": 100 }
        },
        {
            "action_name": "get_networktables_value.py",
            "action_params": {
                "network_table_key": "/SmartDashboard/TargetDistance"
            },
            "position": { "x": 300, "y": 200 }
        },
        {
            "action_name": "detect_apriltags.py",
            "action_params": {
                "families": "tag36h11"
            },
            "position": { "x": 500, "y": 150 }
        },
        {
            "action_name": "pnp_camera_localization.py",
            "action_params": {
                "camera_parameters_path": "/path/to/camera_parameters.json",
                "apriltag_map_path": "/path/to/apriltag_map.json"
            },
            "position": { "x": 700, "y": 180 }
        }
    ]
}
```

In this example, the `GetNetworktablesValue` operation reads a target distance from NetworkTables. This value could then be used by downstream operations to adjust detection parameters, filter results, or modify output behavior.

### Live Configuration Updates

The `network_table_key` parameter can be updated live (without restarting the pipeline) via the WebUI or API.
